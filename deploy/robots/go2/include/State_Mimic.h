#pragma once

#include <atomic>
#include <limits>

#include "FSM/State_RLBase.h"

class State_Mimic : public FSMState
{
public:
    State_Mimic(int state_mode, std::string state_string);

    void enter();
    void run();

    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable()) {
            policy_thread.join();
        }
    }

    class MotionLoader_;
    static std::shared_ptr<MotionLoader_> motion;

private:
    enum class ControlPhase
    {
        EntryVelocity,
        HandoffToMimic,
        Mimic,
        HandoffToVelocity,
    };

    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> velocity_env_;
    std::shared_ptr<MotionLoader_> motion_;
    unitree::common::UnitreeJoystick transition_joystick_;

    std::thread policy_thread;
    bool policy_thread_running = false;
    std::atomic<bool> policy_ready_{false};
    std::atomic<bool> motion_finished_{false};
    std::atomic<ControlPhase> control_phase_{ControlPhase::EntryVelocity};
    std::atomic<float> action_blend_{0.0f};
    Eigen::VectorXf transition_start_joint_pos_;
    float entry_duration_ = 2.0f;
    float handoff_duration_ = 0.5f;
    float exit_duration_ = 1.5f;
    float entry_velocity_ = 1.0f;
    float exit_velocity_ = 0.0f;
    // The shadow velocity policy is only run while the motion's own reference
    // speed is <= velocity_safe_speed_ (its trained envelope). velocity_safe_time_
    // is the motion timestamp after which that holds true all the way to time_range_[1].
    float velocity_safe_speed_ = 1.0f;
    float velocity_safe_time_ = 0.0f;
    bool velocity_safe_range_logged_ = false; // one-shot log only, not control logic
    bool diagnostics_ = false;
    int cycle_frames_ = 27;
    std::array<float, 2> time_range_;
};

class State_Mimic::MotionLoader_
{
public:
    MotionLoader_(std::string motion_file, float fps)
    : dt(1.0f / fps)
    {
        auto data = isaaclab::load_csv(motion_file);

        num_frames = data.size();
        duration = num_frames * dt;

        for (int i = 0; i < num_frames; ++i)
        {
            root_positions.push_back(Eigen::VectorXf::Map(data[i].data(), 3));
            root_quaternions.push_back(Eigen::Quaternionf(data[i][6], data[i][3], data[i][4], data[i][5]));
            dof_positions.push_back(Eigen::VectorXf::Map(data[i].data() + 7, data[i].size() - 7));
        }
        dof_velocities = compute_raw_derivative(dof_positions);

        update(0.0f);
    }

    void update(float time)
    {
        const float last_frame_time = (num_frames - 1) * dt;
        const float clamped_time = std::clamp(time, 0.0f, last_frame_time);
        const float frame = clamped_time / dt;
        index_0_ = std::min(static_cast<int>(std::floor(frame)), num_frames - 1);
        index_1_ = std::min(index_0_ + 1, num_frames - 1);
        blend_ = index_0_ == index_1_ ? 0.0f : frame - index_0_;
    }

    void reset(const isaaclab::ArticulationData & data, float t = 0.0f)
    {
        update(t);
        auto init_to_anchor = isaaclab::yawQuaternion(this->root_quaternion()).toRotationMatrix();
        auto world_to_anchor = isaaclab::yawQuaternion(data.root_quat_w).toRotationMatrix();
        world_to_init_ = world_to_anchor * init_to_anchor.transpose();
    }

    Eigen::VectorXf joint_pos()
    {
        return dof_positions[index_0_] * (1 - blend_) + dof_positions[index_1_] * blend_;
    }

    Eigen::VectorXf root_position()
    {
        return root_positions[index_0_] * (1 - blend_) + root_positions[index_1_] * blend_;
    }

    Eigen::VectorXf joint_vel()
    {
        return dof_velocities[index_0_] * (1 - blend_) + dof_velocities[index_1_] * blend_;
    }

    Eigen::Quaternionf root_quaternion()
    {
        return root_quaternions[index_0_].slerp(blend_, root_quaternions[index_1_]);
    }

    float closest_cycle_time(
        const isaaclab::ArticulationData & data,
        const std::vector<float> & joint_ids_map,
        int cycle_frames,
        float & best_score
    ) const
    {
        const int frames = std::min(cycle_frames, num_frames);
        int best_frame = 0;
        best_score = std::numeric_limits<float>::max();

        for (int frame = 0; frame < frames; ++frame)
        {
            float score = 0.0f;
            for (int i = 0; i < data.joint_pos.size(); ++i)
            {
                const int sdk_id = static_cast<int>(joint_ids_map[i]);
                const float pos_error = (data.joint_pos[i] - dof_positions[frame][sdk_id]) / 0.5f;
                const float vel_error = (data.joint_vel[i] - dof_velocities[frame][sdk_id]) / 5.0f;
                score += pos_error * pos_error + vel_error * vel_error;
            }
            score /= data.joint_pos.size();

            if (score < best_score)
            {
                best_score = score;
                best_frame = frame;
            }
        }

        return best_frame * dt;
    }

    // Horizontal root speed near `frame`, estimated via central finite difference.
    // Used so the exit hand-off can target the motion's *actual* ending speed
    // instead of blindly assuming the clip ends at cruising speed.
    float speed_at(int frame) const
    {
        const int f0 = std::max(frame - 1, 0);
        const int f1 = std::min(frame + 1, num_frames - 1);
        if (f1 <= f0) {
            return 0.0f;
        }
        const Eigen::Vector2f dp = (root_positions[f1] - root_positions[f0]).head<2>();
        return dp.norm() / ((f1 - f0) * dt);
    }

    float dt;
    int num_frames;
    float duration;

    std::vector<Eigen::VectorXf> root_positions;
    std::vector<Eigen::Quaternionf> root_quaternions;
    std::vector<Eigen::VectorXf> dof_positions;
    std::vector<Eigen::VectorXf> dof_velocities;

    Eigen::Matrix3f world_to_init_;

private:
    int index_0_;
    int index_1_;
    float blend_;

    std::vector<Eigen::VectorXf> compute_raw_derivative(const std::vector<Eigen::VectorXf>& data)
    {
        std::vector<Eigen::VectorXf> derivative;
        for (int i = 0; i < data.size() - 1; ++i) {
            derivative.push_back((data[i + 1] - data[i]) / dt);
        }
        derivative.push_back(derivative.back());
        return derivative;
    }
};

REGISTER_FSM(State_Mimic)
