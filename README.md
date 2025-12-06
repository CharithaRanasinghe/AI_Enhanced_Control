# AI_Enhanced_Control
Instead of using traditional control, the DC motor cold start is simulated with adaptive learning, where the system learns from the first part of reaching the target, to sustain the remaining part, finally reaching the target accurately. It is compared with traditional PID control too.

---

## Overview

Traditional PID controllers are effective but often limited in speed and adaptability. By leveraging AI:

- The system **collects data from PID operation** during an initial phase.  
- A **linear regression model** learns the mapping from `(current angle, target)` → `control voltage`.  
- AI predictions are combined with PID residuals, resulting in **improved control performance**.  

This approach illustrates **learning from demonstrations** in control systems, bridging classical techniques and modern AI.

---

## Features

- **DC Motor Simulation:** Models angular motion with PID control.  
- **Online AI Feedforward:**  
  - Collects PID data during initial movement.  
  - Trains AI in real-time.  
  - Combines AI predictions with PID for smoother control.  
- **Interactive Visualization:** Adjustable sliders for:  
  - PID gains (`Kp`, `Ki`, `Kd`)  
  - Target angle  
  - Learning ratio (fraction of initial PID data collection)  
- **Performance Comparison:** Pure PID vs AI-enhanced PID plotted together.

---

## Installation

```bash
git clone https://github.com/CharithaRanasinghe/AI_Enhanced_Control.git
cd AI_Enhanced_Control
pip install numpy matplotlib scikit-learn ipywidgets
AI_Enhanced_Control.py
