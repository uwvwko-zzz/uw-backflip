# Dog MuJoCo sim2sim

Run the 60-D dog actor from the repository root:

    python mujoco/dog/play_onnx.py \
      --onnx=outputs/dog_v1/model_5000_actor.onnx

SPACE triggers one backflip, R performs an explicit physical reset and ESC
exits. The end of the two-second phase does not reset physics.

Headless deterministic validation:

    python mujoco/dog/play_onnx.py \
      --onnx=outputs/dog_v1/model_5000_actor.onnx \
      --headless --no-realtime --start --duration=5

The script uses resources/robots/dog/xml/dog_1.xml and overrides its control
parameters at runtime to match Isaac Gym: FL/FR/RL/RR policy order, 5-ms
physics, 50-Hz actor, fixed
20-ms nominal action delay, Kp/Kd 40/1, action scale 0.5, mirrored joint
coordinates, URDF limits and the 17/34-Nm torque-speed envelope.

The final report separates motion validation from safety diagnostics. A
successful rotation and stable recovery can pass while overspeed or joint-limit
events remain visible as a safety warning.
