# 05. Decoupled Visualization, Camera & Interactive Modes

**Navigation:** [Index](README.md) | **Previous:** [04. Agent Architecture & Training](04_agent_architecture_and_training.md)

---

## 1. Strict One-Way Observation Boundary

The 2D viewer is a **strictly read-only observer** of simulation state. It never steps physics, never modifies agent positions, and is never imported by the training pipeline.

During visual inspection or evaluation, the simulator runs normally and exposes a lightweight snapshot of world geometry, agent poses, raycast endpoints, and objective states (in a GPU-accelerated JAX engine, this simply means copying a single arena's state arrays to host memory once per rendered frame). Because the boundary is strictly one-way, swapping the underlying simulation engine does not break the renderer.

---

## 2. Virtual 2D Camera

A virtual camera maps continuous world coordinates $(w_x, w_y)$ onto pixel coordinates $(s_x, s_y)$ inside a display window of size $(W, H)$:

$$\text{screen}_x = (w_x - c_x) \cdot z + \frac{W}{2}, \qquad \text{screen}_y = (w_y - c_y) \cdot z + \frac{H}{2}$$

where $(c_x, c_y)$ is the camera center in world space and $z$ is the zoom factor (clamped to `[0.2, 4.0]`).

The camera supports three intuitive controls:
- **Mouse-Wheel Zoom:** Multiplies zoom by `1.1×` or `0.9×`.
- **Right-Click Drag Panning:** Shifts $(c_x, c_y)$ inversely by screen drag delta divided by $z$, so panning feels 1:1 at every zoom level.
- **Smooth Agent Following:** Exponentially interpolates the camera center toward a tracked agent's position (`c += (target - c) * 0.1` per frame), which can be toggled on or off with a hotkey.

---

## 3. Layered Rendering Order & Visual Conventions

To keep overlapping elements readable—especially when debugging raycast occlusion and team movement—every frame is drawn in a fixed **5-layer order** on a dark slate background `(24, 28, 36)` using an alpha-blended overlay surface for translucent zones and sensor beams:

1. **Objective / Flag / Base Zones:** Translucent circular fills (`(52, 152, 219, 90)` for neutral goals; team-tinted in CTF) on the alpha overlay paired with crisp solid borders on the main canvas.
2. **Arena Walls & Obstacles:** Solid light-gray line segments `(180, 190, 205)` for outer boundaries, internal walls, and decomposed boxes.
3. **Panoramic Sensor Rays (Toggleable):** Semi-transparent red beams `(231, 76, 60, 60)` on the alpha overlay with solid hit dots `(231, 76, 60)` at each detected intersection endpoint.
4. **Agents & Indicators:** Agent collision circles (`(46, 204, 113)` in solo mode; Red vs. Blue team colors in CTF), yellow forward heading vectors `(241, 196, 15)` of length `radius + 8 px`, and optional communication/flag-carrier indicators.
5. **HUD Telemetry:** Alpha overlay composited onto the main canvas, topped by monospace status lines spaced `18 px` apart (live FPS, zoom level, active mode, arena name, polar distance/bearing, and shortcut hints).

---

## 4. Interactive Modes & Controls

The interactive runner supports two complementary workflows on the exact same physics and camera pipeline:

- **Human Control Mode:** Maps keyboard inputs directly to motor commands, with camera follow disabled by default. Ideal for verifying new arena geometries, checking narrow corridor clearances, and inspecting sliding wall collisions by hand.
- **AI Watch Mode:** Loads a trained policy checkpoint and runs **deterministic inference** (`argmax` over policy logits) at the **10 Hz training cadence**—querying the network once every `6` physics frames and holding that action across the intermediate 60 Hz frames so visual playback matches training dynamics identically.

| Input | Action |
| :--- | :--- |
| `W` / `S` | Move Forward / Backward (Human mode) |
| `A` / `D` | Strafe Left / Right (Human mode) |
| `Q` / `E` or `Left` / `Right` | Rotate Heading Left / Right (Human mode) |
| `F1` | Toggle raycast beam & hit-dot overlay |
| `F2` | Reset arena & zero recurrent hidden state |
| `C` | Toggle smooth camera follow on/off |
| `Right-Click + Drag` / `Wheel` | Pan camera / Zoom in and out |
| `ESC` | Exit viewer |

> [!IMPORTANT]
> **Resetting Recurrent Memory on Replay:** Whenever an episode reaches its goal/end or the user triggers a manual reset (`F2`), the AI Watch loop **must zero out the agent's recurrent hidden state** (`GRU` memory). Carrying stale hidden state across a reset causes the agent to hallucinate walls from the previous run.

---

**Navigation:** [Index](README.md) | **Previous:** [04. Agent Architecture & Training](04_agent_architecture_and_training.md)
