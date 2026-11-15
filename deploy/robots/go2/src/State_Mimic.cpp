#include "State_Mimic.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

static Eigen::Quaternionf init_quat;
std::shared_ptr<State_Mimic::MotionLoader_> State_Mimic::motion = nullptr;

Eigen::Quaternionf base_quat_w(isaaclab::ManagerBasedRLEnv* env)
{
    return env->robot->data.root_quat_w;
}

Eigen::Quaternionf anchor_quat_w(std::shared_ptr<State_Mimic::MotionLoader_> loader)
{
    return loader->root_quaternion();
}

namespace isaaclab
{
namespace mdp
{

REGISTER_OBSERVATION(motion_joint_pos)
{
    auto & robot = env->robot;
    auto & loader = State_Mimic::motion;
    auto & ids = robot->data.joint_ids_map;

    auto data_dfs = loader->joint_pos();
    Eigen::VectorXf data_bfs = Eigen::VectorXf::Zero(data_dfs.size());
    for (int i = 0; i < data_dfs.size(); ++i) {
        data_bfs(i) = data_dfs[ids[i]];
    }
    return std::vector<float>(data_bfs.data(), data_bfs.data() + data_bfs.size());
}

REGISTER_OBSERVATION(motion_joint_vel)
{
    auto & robot = env->robot;
    auto & loader = State_Mimic::motion;
    auto & ids = robot->data.joint_ids_map;

    auto data_dfs = loader->joint_vel();
    Eigen::VectorXf data_bfs = Eigen::VectorXf::Zero(data_dfs.size());
    for (int i = 0; i < data_dfs.size(); ++i) {
        data_bfs(i) = data_dfs[ids[i]];
    }
    return std::vector<float>(data_bfs.data(), data_bfs.data() + data_bfs.size());
}

REGISTER_OBSERVATION(motion_command)
{
    auto & robot = env->robot;
    auto & loader = State_Mimic::motion;
    auto & ids = robot->data.joint_ids_map;

    auto pos_dfs = loader->joint_pos();
    Eigen::VectorXf pos_bfs = Eigen::VectorXf::Zero(pos_dfs.size());
    for (int i = 0; i < pos_dfs.size(); ++i) {
        pos_bfs(i) = pos_dfs[ids[i]];
    }

    auto vel_dfs = loader->joint_vel();
    Eigen::VectorXf vel_bfs = Eigen::VectorXf::Zero(vel_dfs.size());
    for (int i = 0; i < vel_dfs.size(); ++i) {
        vel_bfs(i) = vel_dfs[ids[i]];
    }

    std::vector<float> data;
    data.insert(data.end(), pos_bfs.data(), pos_bfs.data() + pos_bfs.size());
    data.insert(data.end(), vel_bfs.data(), vel_bfs.data() + vel_bfs.size());
    return data;
}

REGISTER_OBSERVATION(motion_anchor_ori_b)
{
    auto real_quat_w = base_quat_w(env);
    auto ref_quat_w = anchor_quat_w(State_Mimic::motion);

    auto rot_relative = (init_quat * ref_quat_w).conjugate() * real_quat_w;
    auto rot = rot_relative.toRotationMatrix().transpose();

    Eigen::Matrix<float, 6, 1> data;
    data << rot(0, 0), rot(0, 1), rot(1, 0), rot(1, 1), rot(2, 0), rot(2, 1);
    return std::vector<float>(data.data(), data.data() + data.size());
}

}
}

State_Mimic::State_Mimic(int state_mode, std::string state_string)
: FSMState(state_mode, state_string)
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());
    auto velocity_policy_dir = param::parser_policy_dir(cfg["velocity_policy_dir"].as<std::string>());

    auto articulation = std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate);
    auto velocity_articulation = std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate);

    std::filesystem::path motion_file = cfg["motion_file"].as<std::string>();
    if (!motion_file.is_absolute()) {
        motion_file = param::proj_dir / motion_file;
    }

    motion_ = std::make_shared<MotionLoader_>(motion_file.string(), cfg["fps"].as<float>());
    spdlog::info("Loaded motion file '{}' with duration {:.2f}s", motion_file.stem().string(), motion_->duration);
    motion = motion_;

    if (cfg["time_start"]) {
        float time_start = cfg["time_start"].as<float>();
        time_range_[0] = std::clamp(time_start, 0.0f, motion_->duration);
    } else {
        time_range_[0] = 0.0f;
    }
    if (cfg["time_end"]) {
        float time_end = cfg["time_end"].as<float>();
        time_range_[1] = std::clamp(time_end, 0.0f, motion_->duration);
    } else {
        time_range_[1] = motion_->duration;
    }

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        articulation
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    velocity_env_ = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(velocity_policy_dir / "params" / "deploy.yaml"),
        velocity_articulation
    );
    velocity_env_->robot->data.joystick = &transition_joystick_;
    velocity_env_->alg = std::make_unique<isaaclab::OrtRunner>(
        velocity_policy_dir / "exported" / "policy.onnx"
    );

    entry_duration_ = cfg["entry_duration"].as<float>(2.0f);
    handoff_duration_ = cfg["handoff_duration"].as<float>(0.5f);
    exit_duration_ = cfg["exit_duration"].as<float>(1.5f);
    entry_velocity_ = cfg["entry_velocity"].as<float>(1.0f);
    cycle_frames_ = cfg["cycle_frames"].as<int>(27);
    diagnostics_ = cfg["diagnostics"].as<bool>(false);

    // The end of `time_range_` is not necessarily cruising speed anymore (e.g. a
    // run clip that already decelerates to a near-stop). Read the motion's own
    // ending speed so the exit hand-off ramps from where the clip actually is,
    // instead of always assuming it needs to decelerate from `entry_velocity_`.
    const int exit_frame = std::min(
        static_cast<int>(time_range_[1] / motion_->dt),
        motion_->num_frames - 1
    );
    exit_velocity_ = cfg["exit_velocity"].as<float>(motion_->speed_at(exit_frame));
    spdlog::info("Mimic exit velocity inferred from motion: {:.3f} m/s", exit_velocity_);

    // Find the latest point in the clip where the reference speed still exceeds
    // what the Velocity policy was trained on; the shadow policy is only allowed
    // to run after that point (see ControlPhase::Mimic).
    velocity_safe_speed_ = cfg["velocity_safe_speed"].as<float>(1.0f);
    {
        const int start_frame = static_cast<int>(time_range_[0] / motion_->dt);
        velocity_safe_time_ = time_range_[0];
        for (int f = exit_frame; f >= start_frame; --f)
        {
            if (motion_->speed_at(f) > velocity_safe_speed_)
            {
                velocity_safe_time_ = std::min((f + 1) * motion_->dt, time_range_[1]);
                break;
            }
        }
    }
    spdlog::info(
        "Velocity shadow policy will stay idle until motion speed <= {:.2f} m/s (from t={:.3f}s of {:.3f}s)",
        velocity_safe_speed_,
        velocity_safe_time_,
        time_range_[1]
    );

    this->registered_checks.emplace_back(
        std::make_pair(
            [this]()->bool { return motion_finished_; },
            FSMStringMap.right.at("Velocity")
        )
    );
    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool { return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_Mimic::enter()
{
    env->robot->update();
    transition_start_joint_pos_ = env->robot->data.joint_pos;
    policy_ready_ = false;
    motion_finished_ = false;
    control_phase_ = ControlPhase::EntryVelocity;
    action_blend_ = 0.0f;
    velocity_safe_range_logged_ = false;
    transition_joystick_.ly(0.0f);
    transition_joystick_.lx(0.0f);
    transition_joystick_.rx(0.0f);

    for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
    {
        lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
        lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
        lowcmd->msg_.motor_cmd()[i].dq() = 0;
        lowcmd->msg_.motor_cmd()[i].tau() = 0;
    }

    motion = motion_;
    env->reset();
    velocity_env_->reset();

    policy_thread_running = true;
    policy_thread = std::thread([this] {
        using clock = std::chrono::high_resolution_clock;
        const std::chrono::duration<double> desired_duration(env->step_dt);
        const auto dt = std::chrono::duration_cast<clock::duration>(desired_duration);

        auto sleep_until = clock::now() + dt;
        float phase_elapsed = 0.0f;
        float motion_time = time_range_[0];
        int log_tick = 0;

        while (policy_thread_running)
        {
            switch (control_phase_.load())
            {
            case ControlPhase::EntryVelocity:
            {
                const float ramp = std::clamp(phase_elapsed / entry_duration_, 0.0f, 1.0f);
                transition_joystick_.ly(entry_velocity_ * ramp);
                velocity_env_->step();
                policy_ready_ = true;
                action_blend_ = std::clamp(phase_elapsed / 0.25f, 0.0f, 1.0f);

                if (phase_elapsed >= entry_duration_)
                {
                    float match_score = 0.0f;
                    motion_time = motion->closest_cycle_time(
                        velocity_env_->robot->data,
                        velocity_env_->robot->data.joint_ids_map,
                        cycle_frames_,
                        match_score
                    );
                    motion->update(motion_time);

                    env->robot->update();
                    auto ref_yaw = isaaclab::yawQuaternion(motion->root_quaternion()).toRotationMatrix();
                    auto robot_yaw = isaaclab::yawQuaternion(base_quat_w(env.get())).toRotationMatrix();
                    init_quat = robot_yaw * ref_yaw.transpose();
                    motion->reset(env->robot->data, motion_time);
                    env->reset();
                    env->step();

                    spdlog::info(
                        "Canter entry matched reference at {:.3f}s with score {:.3f}",
                        motion_time,
                        match_score
                    );
                    control_phase_ = ControlPhase::HandoffToMimic;
                    action_blend_ = 0.0f;
                    phase_elapsed = 0.0f;
                }
                break;
            }
            case ControlPhase::HandoffToMimic:
                transition_joystick_.ly(entry_velocity_);
                velocity_env_->step();
                motion_time += env->step_dt;
                motion->update(motion_time);
                env->step();
                action_blend_ = std::clamp(phase_elapsed / handoff_duration_, 0.0f, 1.0f);
                if (phase_elapsed >= handoff_duration_) {
                    control_phase_ = ControlPhase::Mimic;
                    action_blend_ = 1.0f;
                    phase_elapsed = 0.0f;
                }
                break;

            case ControlPhase::Mimic:
            {
                // The shadow velocity policy never actually drives the motors while
                // it's stepped here, so its `last_action` observation -- which feeds
                // its own previous output back in as input -- has no grounding in
                // physical reality. Left running continuously, that open loop drifts:
                // it doesn't even need an out-of-distribution gait to snowball into
                // saturated garbage (e.g. +/-100 rad), it can happen within ~0.5s even
                // at a normal commanded speed. So we never let it accumulate history:
                // every time we want its opinion, reset it fresh (last_action=0) and
                // take exactly one inference against the robot's real current state --
                // and we only bother doing that once the motion's own speed is back
                // within what it was actually trained on (velocity_safe_speed_).
                const bool in_safe_range = motion_time >= velocity_safe_time_;
                if (in_safe_range)
                {
                    if (!velocity_safe_range_logged_)
                    {
                        velocity_safe_range_logged_ = true;
                        spdlog::info(
                            "Velocity shadow policy resumed at t={:.3f}s (motion back within its trained speed range)",
                            motion_time
                        );
                    }
                    const float span = std::max(time_range_[1] - velocity_safe_time_, 1e-3f);
                    const float progress = std::clamp((motion_time - velocity_safe_time_) / span, 0.0f, 1.0f);
                    transition_joystick_.ly(velocity_safe_speed_ * (1.0f - progress) + exit_velocity_ * progress);
                    velocity_env_->reset();
                    velocity_env_->step();
                }
                motion_time += env->step_dt;
                motion->update(motion_time);
                env->step();
                if (motion_time >= time_range_[1]) {
                    control_phase_ = ControlPhase::HandoffToVelocity;
                    action_blend_ = 0.0f;
                    phase_elapsed = 0.0f;
                    spdlog::info("Canter motion finished; handing control back to Velocity");
                }
                break;
            }

            case ControlPhase::HandoffToVelocity:
            {
                // Decelerate from the motion's actual ending speed, not entry_velocity_:
                // for a clip that already comes to a near-stop, entry_velocity_ would
                // command a sudden "sprint" right as we start handing back control.
                const float deceleration = std::clamp(phase_elapsed / exit_duration_, 0.0f, 1.0f);
                transition_joystick_.ly(exit_velocity_ * (1.0f - deceleration));
                // Same single-shot reset-then-infer pattern as in ControlPhase::Mimic
                // above: never let the shadow policy accumulate its own action history.
                velocity_env_->reset();
                velocity_env_->step();
                motion->update(time_range_[1]);
                env->step();
                // Blend the actual motor targets over the same exit_duration_ window as
                // the speed ramp above (was incorrectly using handoff_duration_, which
                // snapped to 100% velocity_action after just 0.5s while the speed target
                // was still ramping down over 1.5s -- a jerky mismatch at the motor level).
                action_blend_ = deceleration;
                if (phase_elapsed >= exit_duration_) {
                    motion_finished_ = true;
                }
                break;
            }
            }

            // Diagnostic trace (~10Hz): tilt, action blend, joystick target, and how far
            // apart the mimic vs. velocity policy actions are, so we can see exactly
            // where in the timeline instability starts.
            if (diagnostics_ && ++log_tick % 5 == 0)
            {
                const auto & pg = velocity_env_->robot->data.projected_gravity_b;
                const float tilt_deg = std::acos(std::clamp(-pg.z(), -1.0f, 1.0f)) * 180.0f / 3.14159265f;
                const char* phase_name = "?";
                switch (control_phase_.load())
                {
                case ControlPhase::EntryVelocity: phase_name = "EntryVelocity"; break;
                case ControlPhase::HandoffToMimic: phase_name = "HandoffToMimic"; break;
                case ControlPhase::Mimic: phase_name = "Mimic"; break;
                case ControlPhase::HandoffToVelocity: phase_name = "HandoffToVelocity"; break;
                }
                float action_diff = -1.0f;
                if (env->action_manager && velocity_env_->action_manager)
                {
                    auto ma = env->action_manager->processed_actions();
                    auto va = velocity_env_->action_manager->processed_actions();
                    action_diff = 0.0f;
                    for (size_t i = 0; i < ma.size() && i < va.size(); ++i)
                    {
                        const float d = ma[i] - va[i];
                        action_diff += d * d;
                    }
                    action_diff = std::sqrt(action_diff);

                    if (action_diff > 5.0f)
                    {
                        std::string ma_str, va_str;
                        for (size_t i = 0; i < ma.size() && i < va.size(); ++i)
                        {
                            ma_str += fmt::format("{:.2f} ", ma[i]);
                            va_str += fmt::format("{:.2f} ", va[i]);
                        }
                        spdlog::warn("[diag] mimic_action = [ {}] ({} elems)", ma_str, ma.size());
                        spdlog::warn("[diag] velocity_action = [ {}] ({} elems)", va_str, va.size());
                    }
                }
                spdlog::info(
                    "[diag] phase={} t={:.2f} blend={:.2f} joy_ly={:.3f} tilt={:.1f}deg |mimic-vel|={:.3f}",
                    phase_name, phase_elapsed, action_blend_.load(), transition_joystick_.ly(), tilt_deg, action_diff
                );
            }

            phase_elapsed += env->step_dt;

            std::this_thread::sleep_until(sleep_until);
            sleep_until += dt;
        }
    });
}

void State_Mimic::run()
{
    if (!policy_ready_) {
        return;
    }

    auto mimic_action = env->action_manager->processed_actions();
    auto velocity_action = velocity_env_->action_manager->processed_actions();
    const float blend = action_blend_.load();
    const auto phase = control_phase_.load();

    for (int i = 0; i < env->robot->data.joint_ids_map.size(); ++i) {
        float target = mimic_action[i];
        if (phase == ControlPhase::EntryVelocity) {
            target = (1.0f - blend) * transition_start_joint_pos_[i] + blend * velocity_action[i];
        } else if (phase == ControlPhase::HandoffToMimic) {
            target = (1.0f - blend) * velocity_action[i] + blend * mimic_action[i];
        } else if (phase == ControlPhase::HandoffToVelocity) {
            target = (1.0f - blend) * mimic_action[i] + blend * velocity_action[i];
        }
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = target;
    }
}
