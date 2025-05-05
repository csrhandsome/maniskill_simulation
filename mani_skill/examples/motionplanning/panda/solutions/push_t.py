import gymnasium as gym
import numpy as np
import sapien
import torch

from mani_skill.envs.tasks import PushTEnv
from mani_skill.examples.motionplanning.panda.motionplanner_stick import PandaStickMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.utils import compute_grasp_info_by_obb, get_actor_obb

def solve(env: PushTEnv, seed=None, debug=False, vis=False, use_stick_planner=False):
    env.reset(seed=seed)
    
    # 根据参数选择使用哪种规划器
    if use_stick_planner:
        planner = PandaStickMotionPlanningSolver(
            env,
            debug=debug,
            vis=vis,
            base_pose=env.unwrapped.agent.robot.pose,
            visualize_target_grasp_pose=vis,
            print_env_info=False,
            joint_vel_limits=0.5,
            joint_acc_limits=0.5,
        )

    env = env.unwrapped
    
    # -------------------------------------------------------------------------- #
    # Get necessary information about T-shaped object and goal
    # -------------------------------------------------------------------------- #
    # 尝试获取T形物体的网格，如果失败则使用直接访问位置
    try:
        tee_obb = get_actor_obb(env.tee)
        approaching = np.array([0, 0, -1])
        target_closing = env.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
        
        grasp_info = compute_grasp_info_by_obb(
            tee_obb,
            approaching=approaching,
            target_closing=target_closing,
            depth=0.03,
        )
        closing, center = grasp_info["closing"], grasp_info["center"]
    except Exception as e:
        print(f"Warning: Could not get T-shaped object OBB: {e}")
        # 使用直接访问位置
        approaching = np.array([0, 0, -1])
        closing = np.array([0, 1, 0])  # 假设Y轴作为闭合方向
        center = env.tee.pose.sp.p
    
    # 确保goal_pos是numpy数组而不是tensor
    goal_pos = env.goal_tee.pose.p
    if torch.is_tensor(goal_pos):
        goal_pos = goal_pos.cpu().numpy()[0]
    
    # 获取目标姿态
    goal_orientation = env.goal_tee.pose.q
    if torch.is_tensor(goal_orientation):
        goal_orientation = goal_orientation.cpu().numpy()[0]
    
    # 创建工具姿势
    tool_pose = env.l_shape_tool.pose.sp.p if hasattr(env, 'l_shape_tool') else center
    try:
        grasp_pose = env.agent.build_grasp_pose(approaching, closing, tool_pose)
    except:
        # 如果build_grasp_pose方法不可用，手动构建姿势
        print("build_grasp_pose方法不可用,手动构建姿势")
        ortho = np.cross(closing, approaching)
        T = np.eye(4)
        T[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
        T[:3, 3] = tool_pose
        grasp_pose = sapien.Pose(T)
    
    offset = sapien.Pose([0.02, 0, 0])
    grasp_pose = grasp_pose * (offset)
    
    # -------------------------------------------------------------------------- #
    # Reach
    # -------------------------------------------------------------------------- #
    # Position to the side of the T-object to prepare for pushing
    reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
    res = planner.move_to_pose_with_screw(reach_pose)
    if res == -1: return res

    # -------------------------------------------------------------------------- #
    # Grasp/Position tool
    # -------------------------------------------------------------------------- #
    res = planner.move_to_pose_with_screw(grasp_pose)
    if res == -1: return res
    
    # -------------------------------------------------------------------------- #
    # Lift tool to safe height
    # -------------------------------------------------------------------------- #
    lift_height = 0.35  
    lift_pose = sapien.Pose(grasp_pose.p + np.array([0, 0, lift_height]))
    lift_pose.set_q(grasp_pose.q)  # Maintain orientation
    res = planner.move_to_pose_with_screw(lift_pose)
    if res == -1: return res

    # -------------------------------------------------------------------------- #
    # 获取T形物体的详细信息用于更精确的推动
    # -------------------------------------------------------------------------- #
    t_pos = env.tee.pose.sp.p
    if torch.is_tensor(t_pos):
        t_pos = t_pos.cpu().numpy()[0]
    
    # 获取T的当前姿态
    t_current_orientation = env.tee.pose.q
    if torch.is_tensor(t_current_orientation):
        t_current_orientation = t_current_orientation.cpu().numpy()[0]
    
    # T形状由水平和垂直两部分组成，获取其基本尺寸信息
    # 主干（垂直部分）和横条（水平部分）的尺寸
    vertical_length = 0.15  # 主干长度
    horizontal_length = 0.2  # 横条长度
    thickness = 0.05  # 厚度
    
    # 工作高度设置
    contact_height = 0.02  # 接触点高度
    
    # -------------------------------------------------------------------------- #
    # 策略一: 第一阶段 - 粗略平移对准
    # -------------------------------------------------------------------------- #
    # 计算主干中心点到目标点的平移方向
    translation_direction = goal_pos - t_pos
    translation_distance = np.linalg.norm(translation_direction)
    if translation_distance > 0.01:  # 只有当距离足够大时才需要平移
        translation_direction_normalized = translation_direction / translation_distance
        
        # 选择主干的后面作为接触点进行推动
        # 假设T的主干沿着-x方向延伸
        contact_offset = np.array([-vertical_length/2 + thickness/2, 0, 0])
        contact_point = t_pos + contact_offset
        
        # 设置接近点，从较远处接近以确保有足够的加速空间
        approach_distance = 0.15
        approach_point = contact_point - approach_distance * translation_direction_normalized
        approach_point[2] = contact_height
        
        # 移动到接近点
        approach_pose = sapien.Pose(p=approach_point)
        approach_pose.set_q(grasp_pose.q)
        res = planner.move_to_pose_with_screw(approach_pose)
        if res == -1: return res
        
        # 接触T形物体
        contact_point[2] = contact_height
        contact_pose = sapien.Pose(p=contact_point)
        contact_pose.set_q(grasp_pose.q)
        res = planner.move_to_pose_with_screw(contact_pose)
        if res == -1: return res
        
        # 平移推动，但不要完全推到目标位置，留出旋转调整的空间
        push_distance = min(0.9 * translation_distance, 0.2)
        push_target = contact_point + push_distance * translation_direction_normalized
        push_target[2] = contact_height
        
        push_pose = sapien.Pose(p=push_target)
        push_pose.set_q(grasp_pose.q)
        res = planner.move_to_pose_with_screw(push_pose)
        if res == -1: return res
        
        # 抬起工具，准备下一阶段
        lift_point = push_target + np.array([0, 0, 0.1])
        lift_pose = sapien.Pose(p=lift_point)
        lift_pose.set_q(grasp_pose.q)
        res = planner.move_to_pose_with_screw(lift_pose)
        if res == -1: return res
    
    # -------------------------------------------------------------------------- #
    # 策略一: 第二阶段 - 旋转精调
    # -------------------------------------------------------------------------- #
    # 获取T的当前位置和朝向（第一阶段推动后的状态）
    t_current_pos = env.tee.pose.sp.p
    if torch.is_tensor(t_current_pos):
        t_current_pos = t_current_pos.cpu().numpy()[0]
    
    t_current_orientation = env.tee.pose.q
    if torch.is_tensor(t_current_orientation):
        t_current_orientation = t_current_orientation.cpu().numpy()[0]
    
    # 分析当前朝向与目标朝向的差异
    # 这里简化为根据四元数计算朝向差异的角度
    dot_product = np.clip(np.dot(t_current_orientation, goal_orientation), -1.0, 1.0)
    angle_diff = np.arccos(2 * dot_product**2 - 1)
    
    # 只有当角度差异足够大时才需要旋转调整
    if abs(angle_diff) > 0.1:  # 约6度
        # 确定是顺时针还是逆时针旋转
        # 这里简化处理，根据观察到的物体朝向决定旋转方向
        # 假设四元数的某个分量可以指示旋转方向
        clockwise = t_current_orientation[3] < goal_orientation[3]
        
        # 选择T横条的一端作为旋转的接触点
        # 根据旋转方向选择接触点
        if clockwise:
            rotation_contact_offset = np.array([0, horizontal_length/2 - thickness/2, 0])
        else:
            rotation_contact_offset = np.array([0, -horizontal_length/2 + thickness/2, 0])
        
        rotation_contact_point = t_current_pos + rotation_contact_offset
        rotation_contact_point[2] = contact_height
        
        # 接近旋转接触点
        approach_distance = 0.1
        rotation_approach_vector = np.array([0, -1 if clockwise else 1, 0])
        rotation_approach_point = rotation_contact_point + approach_distance * rotation_approach_vector
        rotation_approach_point[2] = contact_height
        
        rotation_approach_pose = sapien.Pose(p=rotation_approach_point)
        rotation_approach_pose.set_q(grasp_pose.q)
        res = planner.move_to_pose_with_screw(rotation_approach_pose)
        if res == -1: return res
        
        # 接触T形物体的横条端点
        rotation_contact_pose = sapien.Pose(p=rotation_contact_point)
        rotation_contact_pose.set_q(grasp_pose.q)
        res = planner.move_to_pose_with_screw(rotation_contact_pose)
        if res == -1: return res
        
        # 旋转推动
        rotation_arc = 0.1  # 旋转弧长
        rotation_push_vector = np.array([1 if clockwise else -1, 0, 0])  # 逆时针推动方向
        rotation_push_point = rotation_contact_point + rotation_arc * rotation_push_vector
        rotation_push_point[2] = contact_height
        
        rotation_push_pose = sapien.Pose(p=rotation_push_point)
        rotation_push_pose.set_q(grasp_pose.q)
        res = planner.move_to_pose_with_screw(rotation_push_pose)
        if res == -1: return res
        
        # 抬起工具，准备最终微调
        lift_point = rotation_push_point + np.array([0, 0, 0.1])
        lift_pose = sapien.Pose(p=lift_point)
        lift_pose.set_q(grasp_pose.q)
        res = planner.move_to_pose_with_screw(lift_pose)
        if res == -1: return res
    
    # -------------------------------------------------------------------------- #
    # 最终微调：确保T形物体准确对齐目标位置
    # -------------------------------------------------------------------------- #
    # 获取当前T的位置（可能在旋转后位置有所偏移）
    t_final_pos = env.tee.pose.sp.p
    if torch.is_tensor(t_final_pos):
        t_final_pos = t_final_pos.cpu().numpy()[0]
    
    # 计算最终需要的平移
    final_translation = goal_pos - t_final_pos
    final_distance = np.linalg.norm(final_translation)
    
    if final_distance > 0.02:  # 只有当偏差足够大时才进行最终推动
        final_translation_normalized = final_translation / final_distance
        
        # 选择主干的后面作为最终推动的接触点
        final_contact_offset = np.array([-vertical_length/2 + thickness/2, 0, 0])
        final_contact_point = t_final_pos + final_contact_offset
        
        # 设置接近点
        final_approach_distance = 0.1
        final_approach_point = final_contact_point - final_approach_distance * final_translation_normalized
        final_approach_point[2] = contact_height
        
        # 移动到接近点
        final_approach_pose = sapien.Pose(p=final_approach_point)
        final_approach_pose.set_q(grasp_pose.q)
        res = planner.move_to_pose_with_screw(final_approach_pose)
        if res == -1: return res
        
        # 接触T形物体
        final_contact_point[2] = contact_height
        final_contact_pose = sapien.Pose(p=final_contact_point)
        final_contact_pose.set_q(grasp_pose.q)
        res = planner.move_to_pose_with_screw(final_contact_pose)
        if res == -1: return res
        
        # 最终推动到目标位置
        final_push_point = final_contact_point + final_distance * final_translation_normalized
        final_push_point[2] = contact_height
        
        final_push_pose = sapien.Pose(p=final_push_point)
        final_push_pose.set_q(grasp_pose.q)
        res = planner.move_to_pose_with_screw(final_push_pose)
        if res == -1: return res
    
    # -------------------------------------------------------------------------- #
    # 完成任务，抬起工具
    # -------------------------------------------------------------------------- #
    # 抬起工具，完成任务
    finish_pose = sapien.Pose(p=final_push_point + np.array([0, 0, 0.1]))
    finish_pose.set_q(grasp_pose.q)
    
    res = planner.move_to_pose_with_screw(finish_pose)
    if res == -1: return res
    
    # -------------------------------------------------------------------------- #
    # Task completion
    # -------------------------------------------------------------------------- #
    planner.close()
    return res