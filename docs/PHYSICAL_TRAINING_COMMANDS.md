lerobot-train \
  --dataset.repo_id=robododo/place-yellow-rectangle_20260702_200850 \
  --policy.type=act \
  --policy.device=cuda \
  --output_dir=outputs/train/place_yellow_rectangle_act_run_20k \
  --job_name=place_yellow_rectangle_act_run_20k \
  --policy.repo_id=robododo/place_yellow_rectangle_act_run_20k \
  --wandb.enable=true \
  --batch_size=8 \
  --steps=20000 \
  --num_workers=4 \
  --policy.chunk_size=56 \
  --policy.n_action_steps=56 \
  --policy.dim_model=256 \
  --policy.dim_feedforward=1024 \
  --policy.n_heads=4 \
  --policy.n_encoder_layers=3 \
  --policy.n_decoder_layers=1 \
  --policy.latent_dim=16 \
  --policy.n_obs_steps=1



## Bind Ports from Window to WSL
usbipd list
## Do this for 2-1, 2-2, 2-6, 2-7
sudo usbipd bind --busid 2-1
usbipd attach --wsl --busid 2-1
## On WSL list out 
sudo apt install usbutils
lsusb


# Install Lerobot on local machine since mapping USB not found
## Just use venv to install 

 .\.lerobot-env\Scripts\Activate.ps1

lerobot-rollout --robot.type=so101_follower --robot.port=COM5 --robot.id=my_awesome_follower_arm --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, overhead: {type: opencv, index_or_path: 2, width: 1920, height: 1080, fps: 30}}" --policy.repo_id=robododo/place_yellow_rectangle_act_run --display_data=true

lerobot-rollout --robot.type=so101_follower --robot.port=COM5 --robot.id=my_awesome_follower_arm --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, overhead: {type: opencv, index_or_path: 2, width: 1920, height: 1080, fps: 30}}" --strategy.type=base --policy.path=robododo/place_yellow_rectangle_act_run_20k --display_data=true

lerobot-rollout --robot.type=so101_follower --robot.port=COM5 --robot.id=my_awesome_follower_arm --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, overhead: {type: opencv, index_or_path: 2, width: 1920, height: 1080, fps: 30}}" --strategy.type=base --policy.path=robododo/place_yellow_rectangle_diffusion_small_v1 --display_data=true


# After Lightbox setup


lerobot-teleoperate --robot.type=so101_follower --robot.port=COM5 --robot.id=my_awesome_follower_arm --teleop.type=so101_leader --teleop.port=COM6 --teleop.id=my_awesome_leader_arm --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640,height: 480, fps: 30}, overhead: {type: opencv, index_or_path: 2, width: 1920, height: 1080, fps: 30}}" --display_data=true

lerobot-record --robot.type=so101_follower --robot.port=COM5 --robot.id=my_awesome_follower_arm --teleop.type=so101_leader --teleop.port=COM6 --teleop.id=my_awesome_leader_arm --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640,height: 480, fps: 30}, overhead: {type: opencv, index_or_path: 2, width: 1920, height: 1080, fps: 30}}" --dataset.repo_id=robododo/place-yellow-rectangle-lightbox --dataset.num_episodes=20 --dataset.single_task="Place Yellow Rectangle in Box" --dataset.streaming_encoding=false --dataset.rgb_encoder.vcodec=h264 --dataset.episode_time_s=15 --dataset.reset_time_s=10 --display_data=true

hf upload robododo/place-yellow-rectangle-lightbox "C:\Users\doris\.cache\huggingface\lerobot\robododo\place-yellow-rectangle-lightbox_20260716_163754" --repo-type=dataset

lerobot-record --robot.type=so101_follower --robot.port=COM5 --robot.id=my_awesome_follower_arm --teleop.type=so101_leader --teleop.port=COM6 --teleop.id=my_awesome_leader_arm --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640,height: 480, fps: 30}, overhead: {type: opencv, index_or_path: 2, width: 1920, height: 1080, fps: 30}}" --dataset.repo_id=robododo/place-yellow-rectangle-lightbox --dataset.num_episodes=20 --dataset.single_task="Place Yellow Rectangle in Box" --dataset.streaming_encoding=false --dataset.rgb_encoder.vcodec=h264 --dataset.episode_time_s=15 --dataset.reset_time_s=10 --display_data=true --resume=true --dataset.root="C:\Users\doris\.cache\huggingface\lerobot\robododo\place-yellow-rectangle-lightbox_20260716_163754" 

Switch to WSL
source ~/venvs/lerobot/bin/activate
mkdir -p ~/datasets

cp -r /mnt/c/Users/doris/.cache/huggingface/lerobot/robododo/place-yellow-rectangle-lightbox_20260716_163754 ~/datasets/

lerobot-train --dataset.repo_id=robododo/place-yellow-rectangle-lightbox --dataset.root="datasets/place-yellow-rectangle-lightbox_20260716_163754" --policy.type=act --policy.device=cuda --policy.repo_id=robododo/place_yellow_rectangle_act_v3 --output_dir=outputs/train/place_yellow_rectangle_act_v3 --job_name=place_yellow_rectangle_act_v3 --policy.chunk_size=50 --policy.n_action_steps=50 --policy.dim_model=256 --policy.dim_feedforward=1024 --policy.n_heads=4 --policy.n_encoder_layers=3 --policy.n_decoder_layers=1 --policy.latent_dim=32 --policy.n_obs_steps=1 --steps=30000 --batch_size=4 --wandb.enable=true

lerobot-rollout --robot.type=so101_follower --robot.port=COM5 --robot.id=my_awesome_follower_arm --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, overhead: {type: opencv, index_or_path: 2, width: 1920, height: 1080, fps: 30}}" --policy.path=robododo/place_yellow_rectangle_act_v3 --display_data=true

## Rollout + record at the same time
lerobot-rollout --robot.type=so101_follower --robot.port=COM5 --robot.id=my_awesome_follower_arm --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, overhead: {type: opencv, index_or_path: 2, width: 1920, height: 1080, fps: 30}}" --policy.path=robododo/place_yellow_rectangle_act_v3 --strategy.type=sentry --strategy.upload_every_n_episodes=4 --dataset.repo_id=robododo/rollout_place_yellow_rectangle_act --dataset.single_task="Rollout for ACT v3 model trained on yellow rectange pick and place" --duration=240
