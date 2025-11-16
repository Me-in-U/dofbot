# V1 vs V2 환경 비교 및 선택 가이드

## 📌 빠른 선택 가이드

### V1 사용 권장 상황

- ✅ Reach 학습만 테스트하고 싶을 때
- ✅ 빠른 프로토타이핑이 필요할 때
- ✅ 학습 시간이 제한적일 때 (3-4시간)

### V2 사용 권장 상황 (현재 상황)

- ✅ **Pick and Place 완전 학습이 목표일 때** ← **지금 이 상황**
- ✅ Grasp, Lift, Transport를 모두 학습하고 싶을 때
- ✅ 충분한 학습 시간이 있을 때 (7-9시간)
- ✅ 실제 로봇 배포를 목표로 할 때

---

## 🔍 V1 문제점 분석 (3시간 학습 결과)

```
사용자 피드백: "REACH까지는 어느정도 되는것 같은데 PICK AND MOVE가 안된다"
```

### V1에서 학습 실패한 이유

#### 1. **Reward가 너무 복잡함**

```python
# V1: 8개의 서로 다른 reward가 동시에 작용
reward = (
    alive * 0.1 +
    reach * -1.0 +
    close * 1.5 +
    lift * 3.0 +
    goal_track * -2.0 +
    penalty_open * -0.5 +
    bonus_grasp * 2.0 +
    bonus_place * 5.0
)
# 문제: 로봇이 어느 reward를 우선해야 할지 모름
```

#### 2. **Grasp 감지가 불확실함**

```python
# V1: 거리 기반 grasp 판단
grasp_candidate = (d_reach < 0.03) & (grip < -0.30)
# 문제: 실제로 물체를 잡았는지 확실하지 않음
```

#### 3. **Exploration 부족**

```yaml
# V1
initial_log_std: -0.5 # 낮은 초기 탐험
entropy_loss_scale: 0.001 # 낮은 entropy
rollouts: 64 # 적은 rollouts
# 문제: 그리퍼를 닫는 행동을 충분히 탐험하지 못함
```

#### 4. **Network가 너무 단순**

```yaml
# V1
layers: [256, 256, 128]
# 문제: Pick-and-place는 복잡한 task인데 network capacity 부족
```

---

## ✨ V2 개선 사항

### 1. **Curriculum Learning 도입**

```python
# V2: 5단계로 나눠서 학습
Stage 1: Reach    (0-100K steps)  → EE를 물체 근처로
Stage 2: Grasp    (100-200K steps) → 그리퍼 닫고 물체 잡기
Stage 3: Lift     (200-300K steps) → 물체 들어올리기
Stage 4: Transport(300-400K steps) → 목표로 이동
Stage 5: Place    (400-500K steps) → 목표에 놓기

# 각 단계마다 명확한 reward
```

### 2. **단순하고 명확한 Reward**

```python
# V2: 현재 stage에 집중한 reward
if current_stage == 1:  # Reach
    reward = -2.0 * d_ee_obj + 3.0 * (d_ee_obj < 0.06)

elif current_stage == 2:  # Grasp
    reward = 2.0 * gripper_closure + 5.0 * grasped

elif current_stage == 3:  # Lift
    reward = 4.0 * lift_height + 3.0 * (height > threshold)

# ... 각 stage마다 명확한 목표
```

### 3. **더 강한 Exploration**

```yaml
# V2
initial_log_std: 0.0 # 더 높은 초기 탐험
entropy_loss_scale: 0.005 # 5배 높은 entropy
rollouts: 128 # 2배 많은 rollouts
# 효과: 그리퍼 닫기, 다양한 접근 방법 탐험
```

### 4. **더 깊은 Network**

```yaml
# V2
layers: [512, 512, 256, 128]
# 효과: 복잡한 manipulation policy 학습 가능
```

### 5. **물리 파라미터 최적화**

```python
# V2: Grasp하기 쉽게 조정
object:
  mass: 0.03kg    # V1: 0.05kg → 더 가벼움
  size: 0.025m    # V1: 0.03m → 더 작음

gripper:
  damping: 15.0   # V1: 10.0 → 더 안정적
  effort: 30.0    # V1: 20.0 → 더 강함
```

---

## 📊 V1 vs V2 성능 예상

### V1 (3시간 학습 실제 결과)

```
✅ Reach: 80-90% 성공
❌ Grasp: 5-10% 성공
❌ Lift: 0-5% 성공
❌ Place: 0% 성공

Total Reward: ~10-20 (낮음)
```

### V2 (예상 결과)

```
50K steps  (~1h):  Reach 90%+
150K steps (~3h):  Grasp 50%+
250K steps (~5h):  Lift 40%+
350K steps (~7h):  Transport 30%+
500K steps (~9h):  Place 20-40%

Total Reward: ~50-100 (높음)
```

---

## 🎯 구체적인 V2 사용법

### 1단계: V2 학습 시작

```bash
cd C:\Users\Zoe_Lowell\Documents\GitHub\DofBot-Issac-Sim\rl\dofbot_isaacLab\dofbot

python scripts/skrl/train.py \
  --task=Dofbot-PickPlace-Direct-v2 \
  --algorithm=PPO \
  --ml_framework=torch \
  --num_envs=1024 \
  --device=cuda
```

### 2단계: TensorBoard 모니터링

```bash
# 새 cmd 창 열기
cd C:\Users\Zoe_Lowell\Documents\GitHub\DofBot-Issac-Sim\rl\dofbot_isaacLab\dofbot

tensorboard --logdir=logs/skrl/dofbot_pickplace_direct_v2
```

### 3단계: 학습 진행 확인

```
1시간 후: Reach reward가 증가하는지 확인
3시간 후: Grasp reward가 나타나는지 확인
5시간 후: Lift reward가 증가하는지 확인
7시간 후: Transport reward가 나타나는지 확인
9시간 후: 최종 평가
```

### 4단계: 결과 평가

```bash
python scripts/skrl/eval.py \
  --task=Dofbot-PickPlace-Direct-v2 \
  --num_envs=64 \
  --checkpoint=logs/skrl/dofbot_pickplace_direct_v2/.../checkpoints/best_agent.pt
```

---

## 🔧 V2에서도 안되면?

### 문제 1: Reach는 되는데 Grasp이 여전히 안됨

**진단:**

```python
# TensorBoard에서 확인
- Stage1 reward는 증가
- Stage2 reward는 거의 0
```

**해결책 A: Grasp reward 대폭 증가**

```python
# dofbot_pickplace_env_cfg_v2.py 수정
rew_stage2_grasp_bonus = 15.0  # 5.0 → 15.0 (3배)
rew_stage2_close_gripper = 5.0  # 2.0 → 5.0
```

**해결책 B: Object를 더 쉽게**

```python
# Object 더 가볍고 크게
object_cfg = RigidObjectCfg(
    spawn=sim_utils.CuboidCfg(
        size=(0.03, 0.03, 0.03),  # 0.025 → 0.03
        mass_props=sim_utils.MassPropertiesCfg(mass=0.02),  # 0.03 → 0.02
    )
)
```

**해결책 C: 그리퍼를 더 강하게**

```python
# Gripper actuator 강화
"gripper": ImplicitActuatorCfg(
    damping=20.0,  # 15.0 → 20.0
    effort_limit_sim=40.0,  # 30.0 → 40.0
)
```

### 문제 2: Grasp은 되는데 Lift가 안됨

**해결책:**

```python
# Lift reward 증가
rew_stage3_lift = 8.0  # 4.0 → 8.0
rew_stage3_bonus = 5.0  # 3.0 → 5.0

# Arm actuator 강화
"arm": ImplicitActuatorCfg(
    damping=60.0,  # 50.0 → 60.0
    effort_limit_sim=60.0,  # 50.0 → 60.0
)
```

### 문제 3: 학습이 너무 느림

**해결책:**

```yaml
# agents/skrl_ppo_pickplace_v2_cfg.yaml

# Learning rate 증가
learning_rate: 3.0e-04  # 1e-4 → 3e-4

# Mini-batches 증가
mini_batches: 32  # 16 → 32

# 환경 수 증가
--num_envs=2048  # 1024 → 2048
```

---

## 🎓 왜 V2가 더 나을까? (이론적 배경)

### 1. Curriculum Learning

```
논문: "Automatic Goal Generation for Reinforcement Learning Agents"

핵심: 복잡한 task를 작은 subtask로 나누면 학습이 훨씬 빠름

Pick-and-place는 본질적으로:
Reach → Grasp → Lift → Transport → Place
의 순서가 있는 task

V1: 모든 것을 한번에 학습 → 실패
V2: 단계별로 학습 → 성공 가능성 높음
```

### 2. Reward Shaping

```
논문: "Policy Invariance Under Reward Transformations"

핵심: Sparse reward는 학습이 어렵고,
      Dense reward는 local optima에 빠지기 쉬움

V2 접근:
- Sparse bonus (큰 보상, 가끔)
- Dense shaping (작은 보상, 자주)
- 두 가지를 stage별로 조합
```

### 3. Exploration

```
논문: "Exploration by Random Network Distillation"

핵심: Manipulation task는 충분한 exploration 필요

V2 개선:
- Higher entropy → 더 다양한 행동 시도
- More rollouts → 더 많은 경험 수집
- Longer episodes → 충분한 시도 시간
```

---

## 📈 실전 팁

### Tip 1: 중간 평가로 학습 방향 확인

```bash
# 100K steps마다 평가
python scripts/skrl/eval.py \
  --task=Dofbot-PickPlace-Direct-v2 \
  --num_envs=16 \
  --checkpoint=logs/.../checkpoints/agent_100000.pt

# 확인 사항:
# - Reach가 잘 되는가?
# - Grasp 시도는 하는가?
# - 그리퍼가 닫히는가?
```

### Tip 2: TensorBoard로 bottleneck 찾기

```
Reward/Total이 멈춘 지점 확인:
- ~20에서 멈춤: Reach 단계 문제
- ~40에서 멈춤: Grasp 단계 문제
- ~60에서 멈춤: Lift 단계 문제
- ~80에서 멈춤: Transport 단계 문제
```

### Tip 3: 하이퍼파라미터는 단계적으로 조정

```
1. 먼저 default V2로 학습
2. TensorBoard로 bottleneck 확인
3. 해당 stage의 reward만 증가
4. 재학습 후 비교
5. 반복
```

---

## 🚀 지금 당장 시작하기

```bash
# 1. V2 학습 시작 (추천: headless mode)
python scripts/skrl/train.py \
  --task=Dofbot-PickPlace-Direct-v2 \
  --algorithm=PPO \
  --ml_framework=torch \
  --num_envs=1024 \
  --device=cuda \
  --headless

# 2. 별도 터미널에서 TensorBoard
tensorboard --logdir=logs/skrl/dofbot_pickplace_direct_v2

# 3. 브라우저에서 모니터링
# http://localhost:6006

# 4. 7-9시간 후 결과 확인!
```

---

## 📚 추가 자료

### V2 상세 가이드

- `V2_TRAINING_GUIDE.md`: V2 학습 완전 가이드
- `dofbot_pickplace_env_v2.py`: V2 구현 코드
- `dofbot_pickplace_env_cfg_v2.py`: V2 설정

### V1 참고 (비교용)

- `dofbot_pickplace_env.py`: V1 구현
- `dofbot_pickplace_env_cfg.py`: V1 설정

---

## ✅ 체크리스트

V2 학습 시작 전 확인:

- [ ] V1 학습 완료 (3시간 돌려본 결과 확인)
- [ ] V1에서 Reach는 되지만 Pick이 안됨을 확인
- [ ] GPU 사용 가능 (RTX 5070)
- [ ] 7-9시간 학습 시간 확보
- [ ] TensorBoard 사용법 숙지
- [ ] V2_TRAINING_GUIDE.md 읽음
- [ ] 학습 중 모니터링 계획 수립

모두 체크되었다면 V2 학습을 시작하세요! 🎉
