# Training an SO101 policy with MuJoCo demonstrations

This project uses **imitation learning**, matching the existing physical ACT
workflow. MuJoCo supplies the robot, objects, physics, and camera images. You
move the physical SO101 leader arm to produce successful actions in simulation.
Those demonstrations become a LeRobot dataset, which can be mixed with the
existing real-robot data and used to fine-tune ACT.

The workflow is:

1. teleoperate MuJoCo and save successful demonstrations;
2. inspect and, if desired, merge the simulation and physical datasets;
3. fine-tune the existing ACT checkpoint on a CUDA GPU;
4. run closed-loop MuJoCo evaluations over several random seeds;
5. deploy to the physical follower only after the simulated camera views and
   motion conventions have been validated.

## 1. Activate the environment

Open **Anaconda Prompt** or PowerShell with Conda initialized:

```powershell
conda activate lerobot
cd C:\Users\doris\Desktop\robotics\sim2real
```

Find the serial port for the SO101 **leader** arm:

```powershell
lerobot-find-port
```

Unplug the leader USB cable, press Enter, reconnect it, and note the reported
port, such as `COM5`.

## 2. Record successful simulation demonstrations

The recorder reads only the leader arm. It does not command the physical
follower. Replace `COM5` with the port found above:

```powershell
python record_sim_teleop.py --port COM5 --episodes 20
```

For every episode:

1. put the leader in a safe starting pose and press Enter;
2. use the camera window to pick up the green or yellow block;
3. place it into the tray with the same color;
4. let the block settle, then save the episode;
5. retry and discard any failed or poor demonstration.

Press `Q` or Escape in the camera window to stop the current episode early.
The default output is:

```text
C:\Users\doris\.cache\huggingface\lerobot\robododo\so101-mujoco-color-sort
```

The script refuses to overwrite that directory. To deliberately start the
simulation dataset again from zero, add `--overwrite`.

Start with 10-20 demonstrations as a pipeline test. For a useful fine-tune,
record at least 50 varied successful episodes. Vary block position, block
color, and initial arm pose while keeping the task physically plausible.

Inspect the completed dataset:

```powershell
lerobot-edit-dataset `
  --repo_id robododo/so101-mujoco-color-sort `
  --root C:/Users/doris/.cache/huggingface/lerobot/robododo/so101-mujoco-color-sort `
  --operation.type info
```

## 3. Merge simulation and physical demonstrations

The recorder deliberately uses the physical dataset's exact schema: 30 Hz,
six joint features, `front` at 640x480, and `overhead` at 1280x720. This allows
LeRobot to merge the two datasets.

```powershell
lerobot-edit-dataset `
  --new_repo_id robododo/place-rectangle-colored-box-real-sim `
  --new_root C:/Users/doris/.cache/huggingface/lerobot/robododo/place-rectangle-colored-box-real-sim `
  --operation.type merge `
  --operation.repo_ids '["robododo/place-rectangle-colored-box","robododo/so101-mujoco-color-sort"]' `
  --operation.roots '["C:/Users/doris/.cache/huggingface/lerobot/robododo/place-rectangle-colored-box_20260718_195758","C:/Users/doris/.cache/huggingface/lerobot/robododo/so101-mujoco-color-sort"]'
```

Mixing the datasets preserves the real camera appearance and robot dynamics.
Training on simulation alone is more likely to overfit to rendered images.

## 4. Fine-tune the existing ACT policy

Resolve the local Hugging Face checkpoint cache to its actual snapshot:

```powershell
$policyCache = 'C:\Users\doris\.cache\huggingface\hub\models--robododo--sort_blocks_by_color_act_v2'
$revision = Get-Content "$policyCache\refs\main"
$basePolicy = "$policyCache\snapshots\$revision"
```

### Recommended on this computer: Hugging Face Jobs

The current `lerobot` environment has CPU-only PyTorch. ACT training will be
impractically slow there, so submit the same training command to a CUDA worker.
Authentication and billing must be configured for the Hugging Face account:

```powershell
hf auth login
hf jobs hardware
```

Then run a short 5,000-step fine-tune first:

```powershell
lerobot-train `
  --dataset.repo_id=robododo/place-rectangle-colored-box-real-sim `
  --dataset.root=C:/Users/doris/.cache/huggingface/lerobot/robododo/place-rectangle-colored-box-real-sim `
  --policy.path="$basePolicy" `
  --policy.device=cuda `
  --policy.repo_id=robododo/sort-blocks-real-sim-act `
  --output_dir=C:/Users/doris/Desktop/robotics/sim2real/outputs/train/act-real-sim `
  --steps=5000 `
  --save_freq=1000 `
  --job.target=a10g-small
```

The command uploads a local-only dataset privately for the remote job and
pushes the final model to `robododo/sort-blocks-real-sim-act`.

### Local CUDA alternative

If CUDA-enabled PyTorch is installed in the environment, remove
`--job.target=a10g-small` from the command. Confirm CUDA first:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA')"
```

## 5. Evaluate the fine-tuned policy in MuJoCo

For a locally produced checkpoint:

```powershell
$trainedPolicy = 'C:\Users\doris\Desktop\robotics\sim2real\outputs\train\act-real-sim\checkpoints\last\pretrained_model'
python run_act_policy_in_mujoco.py --policy-path "$trainedPolicy" --seed 0 --steps 500
```

The evaluator writes `mujoco_act_rollout.mp4`. Test multiple block colors and
spawn positions by changing the seed:

```powershell
0..9 | ForEach-Object {
  python run_act_policy_in_mujoco.py `
    --policy-path "$trainedPolicy" `
    --seed $_ `
    --steps 500 `
    --output "mujoco_act_rollout_seed_$_.mp4"
}
```

Count success only when the block reaches the matching tray and stays there.
A ten-seed run is a smoke test; use at least 50 fixed seeds when comparing
checkpoints.

## What simulation training can and cannot establish

The data pipeline is compatible with the physical policy, but useful transfer
depends on calibration. Before trusting a simulation-trained checkpoint on the
real follower, compare these items against a few physical frames:

- wrist and overhead camera position, field of view, orientation, and crop;
- leader/follower joint zero points and direction signs;
- gripper opening (`0-100`) versus the MuJoCo finger geometry;
- block and tray scale, mass, friction, and contact behavior;
- lighting, background, and color distribution.

The next improvement after the first successful pipeline run is domain
randomization: vary light intensity and color, camera pose by a few millimeters
and degrees, object friction/mass, and block spawn pose. Keep a fixed evaluation
seed set so randomization changes can be compared fairly.
