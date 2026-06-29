from .trainer_factory import register_trainer, make_trainer
import trainers.vec_obs.single_env_training.ppo_vec, trainers.vis_obs.single_env_training.ppo_vis
import trainers.vec_obs.single_env_training.bc_vec, trainers.vis_obs.single_env_training.bc_vis
import trainers.vec_obs.multi_env_training.mult_env_vec, trainers.vis_obs.multi_env_training.mult_env_vis

