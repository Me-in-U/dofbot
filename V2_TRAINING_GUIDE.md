# Dofbot Pick-and-Place V2 Training Guide

## 🎯 V1 vs V2 주요 차이점

### V1 문제점 분석

- ✅ **Reach 단계**: 학습됨
- ❌ **Pick 단계**: 학습 안됨
- ❌ **Move 단계**: 학습 안됨

### V2 개선 사항

#### 1. **Curriculum Learning (단계적 학습)**

```python
Stage 1: Reach    → EE를 물체에 접근
Stage 2: Grasp    → 그리퍼로 물체 잡기
Stage 3: Lift     → 물체를 테이블에서 들어올리기
Stage 4: Transport→ 잡은 물체를 목표로 이동
Stage 5: Place    → 목표 위치에 놓기
```

#### 2. **더 명확한 Reward 구조**

```yaml
V1 (복잡한 reward):
  - 8개의 서로 다른 reward 항목
  - 상호작용이 불명확

V2 (단계별 reward):
  - 각 stage마다 명확한 reward
  - 진행 상황에 따라 자동으로 다음 stage로 전환
  - Sparse bonus + Dense shaping 조합
```

#### 3. **Policy Network 개선**

```yaml
V1: [256, 256, 128]
V2: [512, 512, 256, 128] # 더 깊은 네트워크
```

#### 4. **더 나은 Exploration**

```yaml
V1:
  - initial_log_std: -0.5
  - entropy_loss_scale: 0.001
  - rollouts: 64

V2:
  - initial_log_std: 0.0 # 초기 탐험 증가
  - entropy_loss_scale: 0.005 # 5배 높은 entropy
  - rollouts: 128 # 2배 많은 rollouts
```

#### 5. **물리 파라미터 개선**

```python
V1:
  - object mass: 0.05kg
  - object size: 0.03m
  - damping: 40.0, 10.0

V2:
  - object mass: 0.03kg      # 더 가벼움
  - object size: 0.025m      # 더 작음
  - damping: 50.0, 15.0      # 더 안정적
```

---

## 🚀 V2 학습 실행

### 기본 학습 (권장)

```bash
python scripts/skrl/train.py \
  --task=Dofbot-PickPlace-Direct-v2 \
  --algorithm=PPO \
  --ml_framework=torch \
  --num_envs=1024 \
  --device=cuda
```

### 더 많은 환경으로 학습 (GPU 여유 있을 때)

```bash
python scripts/skrl/train.py \
  --task=Dofbot-PickPlace-Direct-v2 \
  --algorithm=PPO \
  --ml_framework=torch \
  --num_envs=2048 \
  --device=cuda
```

### Headless 모드 (더 빠른 학습)

```bash
python scripts/skrl/train.py \
  --task=Dofbot-PickPlace-Direct-v2 \
  --algorithm=PPO \
  --ml_framework=torch \
  --num_envs=1024 \
  --device=cuda \
  --headless
```

---

## 📊 학습 모니터링

### TensorBoard 실행

```bash
# 새 터미널 열기
cd C:\Users\Zoe_Lowell\Documents\GitHub\DofBot-Issac-Sim\rl\dofbot_isaacLab\dofbot

tensorboard --logdir=logs/skrl/dofbot_pickplace_direct_v2
```

브라우저에서 `http://localhost:6006` 접속

### 주요 모니터링 지표

#### 1. **Stage 진행 상황** (추가 구현 필요)

학습이 각 stage를 얼마나 잘 진행하는지 확인

#### 2. **Reward 항목별 분석**

```
- Reward/Total: 전체 보상 (점진적 증가 기대)
- Stage1/Reach: Reach 성공률
- Stage2/Grasp: Grasp 성공률
- Stage3/Lift: Lift 성공률
- Stage4/Transport: Transport 진행도
- Stage5/Place: Place 성공률
```

#### 3. **Policy Loss**

- 안정적으로 감소해야 함
- 급격한 변화는 learning rate 문제

#### 4. **Episode Length**

- 초기: ~900 (15초 \* 60 FPS)
- 학습 후: 점점 짧아짐 (빠른 성공)

---

## 🎓 Curriculum Learning 작동 방식

### Stage 자동 전환

```python
# 환경 내부에서 자동으로 stage 추적
self._current_stage[env_i]:
  - 0: 아직 도달 안함
  - 1: 도달 완료 (reached)
  - 2: 잡기 완료 (grasped)
  - 3: 들어올림 완료 (lifted)
  - 4: 목표 근처 (near_goal)
```

### Reward 변화

```python
# Stage 1: 주로 reach reward
reward = -2.0 * d_ee_obj + 3.0 * reached

# Stage 2: grasp reward 추가
reward += 2.0 * gripper_closure + 5.0 * grasped

# Stage 3: lift reward 추가
reward += 4.0 * lift_progress + 3.0 * lifted

# Stage 4: transport reward 추가
reward += -1.5 * d_obj_goal + 2.0 * goal_proximity

# Stage 5: place bonus
reward += 10.0 * placed
```

---

## 🔧 하이퍼파라미터 튜닝 가이드

### 학습이 너무 느리면

```yaml
# agents/skrl_ppo_pickplace_v2_cfg.yaml

# 1. Learning rate 증가
learning_rate: 3.0e-04  # 1e-4 → 3e-4

# 2. Rollouts 증가
rollouts: 256  # 128 → 256

# 3. 환경 수 증가
--num_envs=2048  # 1024 → 2048
```

### Grasp이 안되면

```python
# dofbot_pickplace_env_cfg_v2.py

# 1. Grasp reward 증가
rew_stage2_grasp_bonus = 10.0  # 5.0 → 10.0

# 2. Grasp threshold 완화
grasp_threshold = 0.05  # 0.04 → 0.05

# 3. Gripper actuator 강화
damping=20.0  # 15.0 → 20.0
```

### Lift가 안되면

```python
# 1. Lift reward 증가
rew_stage3_lift = 6.0  # 4.0 → 6.0

# 2. Object를 더 가볍게
mass=0.02  # 0.03 → 0.02

# 3. Lift threshold 낮춤
lift_threshold = 0.06  # 0.08 → 0.06
```

### Policy가 불안정하면

```yaml
# 1. Learning rate 감소
learning_rate: 5.0e-05 # 1e-4 → 5e-5

# 2. Gradient clipping 강화
grad_norm_clip: 0.3 # 0.5 → 0.3

# 3. Entropy 감소
entropy_loss_scale: 0.002 # 0.005 → 0.002
```

---

## 📈 예상 학습 시간

### RTX 5070 기준 (1024 envs)

```
Total timesteps: 500,000
FPS: ~15-20 (V2는 더 복잡한 연산)

예상 시간: ~7-9시간
Checkpoint 간격: 50,000 steps (~1시간)
```

### 학습 단계별 예상 결과

```
50K steps  (~1h):  Reach 학습 완료
150K steps (~3h):  Grasp 학습 시작
250K steps (~5h):  Lift 학습 완료
350K steps (~7h):  Transport 학습 시작
500K steps (~9h):  Place 성공률 30-50%
```

---

## 🎯 성공 기준

### 최소 성공 기준

```
- Reach success: >90%
- Grasp success: >60%
- Lift success: >40%
- Place success: >20%
```

### 목표 성공 기준

```
- Reach success: >95%
- Grasp success: >80%
- Lift success: >60%
- Place success: >40%
```

---

## 🐛 Troubleshooting

### 문제: "Reach는 되는데 Grasp이 안됨"

**해결책:**

1. `rew_stage2_grasp_bonus` 증가 (5.0 → 10.0)
2. `grasp_threshold` 증가 (0.04 → 0.06)
3. 그리퍼 초기 위치를 약간 닫힌 상태로 (`grip_joint: -0.5`)

### 문제: "Grasp은 되는데 Lift가 안됨"

**해결책:**

1. Object mass 감소 (0.03 → 0.02)
2. `rew_stage3_lift` 증가 (4.0 → 6.0)
3. Arm actuator effort limit 증가 (50 → 60)

### 문제: "Lift는 되는데 Transport가 안됨"

**해결책:**

1. `rew_stage4_progress` 증가 (2.0 → 4.0)
2. Episode length 증가 (15초 → 20초)
3. Goal-object separation 감소 (0.25 → 0.20)

### 문제: "학습이 전혀 안됨"

**해결책:**

1. V1으로 돌아가서 Reach부터 다시 확인
2. `initial_log_std` 증가 (0.0 → 0.5) - 더 많은 exploration
3. Learning rate 감소 (1e-4 → 5e-5) - 안정성 증가

---

## 📁 저장된 모델 위치

```
logs/skrl/dofbot_pickplace_direct_v2/
└── YYYY-MM-DD_HH-MM-SS_ppo_torch/
    ├── checkpoints/
    │   ├── agent_50000.pt
    │   ├── agent_100000.pt
    │   ├── ...
    │   └── best_agent.pt
    ├── runs/
    └── config.yaml
```

---

## 🎬 V2 평가 실행

```bash
# 최고 모델 평가
python scripts/skrl/eval.py \
  --task=Dofbot-PickPlace-Direct-v2 \
  --num_envs=64 \
  --checkpoint=logs/skrl/dofbot_pickplace_direct_v2/YYYY-MM-DD_HH-MM-SS_ppo_torch/checkpoints/best_agent.pt

# 특정 체크포인트 평가
python scripts/skrl/eval.py \
  --task=Dofbot-PickPlace-Direct-v2 \
  --num_envs=64 \
  --checkpoint=logs/skrl/dofbot_pickplace_direct_v2/YYYY-MM-DD_HH-MM-SS_ppo_torch/checkpoints/agent_500000.pt
```

---

## 📚 참고 자료

### V2 설계 기반 논문

- **Curriculum Learning**: "Automatic Curriculum Learning For Deep RL"
- **Reward Shaping**: "Policy Invariance Under Reward Transformations"
- **Manipulation Learning**: "Learning Dexterous Manipulation from Suboptimal Experts"

### V1 대비 V2 변경사항 요약

| 항목           | V1              | V2                | 이유             |
| -------------- | --------------- | ----------------- | ---------------- |
| Reward 구조    | 복잡한 8개 항목 | 5단계 curriculum  | 명확한 학습 목표 |
| Network        | [256,256,128]   | [512,512,256,128] | 복잡한 task 대응 |
| Exploration    | entropy=0.001   | entropy=0.005     | Grasp 탐색 증가  |
| Rollouts       | 64              | 128               | 안정적 학습      |
| Training steps | 200K            | 500K              | Curriculum 완료  |
| Object mass    | 0.05kg          | 0.03kg            | Grasp 용이       |
| Episode length | 10s             | 15s               | 충분한 시간      |

---

## ✨ V2 사용 시작하기

```bash
# 1. V2 환경 학습 시작
python scripts/skrl/train.py --task=Dofbot-PickPlace-Direct-v2 --num_envs=1024 --device=cuda

# 2. 별도 터미널에서 TensorBoard 실행
tensorboard --logdir=logs/skrl/dofbot_pickplace_direct_v2

# 3. 학습 모니터링
# - http://localhost:6006 접속
# - Total reward가 점진적으로 증가하는지 확인
# - Policy loss가 안정적인지 확인

# 4. 7-9시간 후 결과 확인
python scripts/skrl/eval.py \
  --task=Dofbot-PickPlace-Direct-v2 \
  --num_envs=64 \
  --checkpoint=logs/skrl/dofbot_pickplace_direct_v2/.../checkpoints/best_agent.pt
```

Good luck! 🚀
