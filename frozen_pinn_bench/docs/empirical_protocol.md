# 실측 갈래: 벤치 로그 형식과 실행

실측 갈래는 벤치에서 기록한 로그를 분석 갈래와 **같은 지표와 같은 그림 형식**으로 바꿉니다. 지표는 RMS 잔차, 스펙트럼, HAPS 등가 BER·데이터율·outage, 획득 constellation입니다. 그림 번호는 E01–E08입니다.

```
data/empirical/<session>/            ← 벤치 로그 (입력)
results/empirical/<session>/         ← summary.json, data.npz, figures/E* (PNG·PDF·EPS) (출력)
```

## 1. 벤치에서 돌릴 제어기

분석 실행(`results/analytical/<run>/models/`)이 실제 벤치에 올릴 제어기를 저장합니다.

| 파일 | 내용 |
|---|---|
| `frozen.npz` | Frozen-PINN 예측기: 고정 은닉층 `W, b`, 출력층 `B₀`, 정규화 `σ₀, σ₁` (`pat.predictor.load_model`) |
| `linear.npz` | M = 0 ablation |
| `command_layer.npz` | 물리 잔차 명령층 이득 `K_u`, collocation 시각 `τ`, 설정 |
| `controllers.json` | 교란군별 PID 이득, 온라인 재적합 설정(W, λ_o, N_r), 명령층 지평 |

`pat.controllers`의 `PIDController`와 `PredictiveController(..., command='pinn')`는 벤치 루프와 같은 인터페이스입니다.
- `reset(context)`: `LoopContext(spec, 1, C_F[None], p_tar[None])`
- `on_frame(reading, capture_step, now)`: PSD 프레임 도착
- `tick(now)`: 2 ms마다 명령 반환(기계각, rad)

벤치 소프트웨어에서 이 객체를 그대로 부르면 시뮬레이션과 같은 제어기가 됩니다. 시간은 0.5 ms 단위 step 정수입니다.

학습 데이터를 벤치에서 새로 얻으려면, FSM을 고정한 채(hold) 레일 프로파일을 돌려 PSD 프레임을 기록하고 같은 `ContextModel.fit`에 넣습니다. 레이블이 미래 PSD 프레임이므로 추가 센서는 필요 없습니다.

## 2. 실험 절차

1. **보정:** FSM을 고정하고 PDA 전력이 최대가 되는 위치를 찾아, 그때의 PSD 좌표를 `p_tar`로 기록합니다. FSM 명령을 x, y로 조금씩 움직여 `J_F`(PSD 이동/명령)를 측정합니다.
2. **프로파일:** 교란군마다 seed를 정하고, 같은 프로파일을 모든 제어기에 똑같이 재생합니다. 권장 시간은 20 s(처음 2 s는 분석에서 제외)이고 seed당 2회 이상입니다.
   - `station`, `fast_sweep`: 레일만 사용합니다.
   - `vibration`: 레일 station + 6–18 Hz 진동을 FSM 명령에 더합니다. 제어기는 진동을 모릅니다.
   - `acquisition`: 레일을 2 mm 오프셋에 세운 뒤 제어를 시작합니다(3 s).
3. **제어기:** 최소 `pid`와 `frozen_pinn`이 필요하고, 선택으로 `hold`, `linear`를 추가합니다. 그림은 `pid`와 `frozen_pinn`을 그립니다.
4. **USRP:** PDA 출력으로 64-QAM을 송수신하며 프레임별 EVM과 BER을 기록합니다(선택).

## 3. 로그 형식

```
<session>/session.json
<session>/runs/<family>_s<seed>__<controller>.json
<session>/runs/<family>_s<seed>__<controller>.npz
```

**session.json**

```json
{
  "date": "2026-10-20",
  "synthetic": false,
  "calibration": {"p_tar_m": [x, y], "J_F": [[..., ...], [..., ...]]},
  "warmup_s": 2.0,
  "notes": "stage, jitter measurement, path lengths ..."
}
```

**runs/*.json**: `{"family": "fast_sweep", "controller": "frozen_pinn", "seed": 9110, "duration_s": 20.0}`

**runs/*.npz** (SI 단위)

| 키 | 모양 | 필수 | 내용 |
|---|---|---|---|
| `frame_t_s` | (K,) | ✓ | PSD 프레임 수신 시각(증가해야 함) |
| `psd_xy_m` | (K, 2) | ✓ | PSD 위치 [m]. CONEX 출력이 mm이면 변환 |
| `command_t_s` | (C,) | ✓ | 명령 시각 |
| `command_rad` | (C, 2) | ✓ | 제어기가 낸 FSM 기계각 [rad], 주입 진동 제외 |
| `rail_t_s`, `rail_m` | (R,) | | 레일 엔코더 위치 |
| `injection_t_s`, `injection_rad` | (I,), (I, 2) | | 주입한 진동(검증용) |
| `psd_power` | (K,) | | PSD 합 신호 |
| `usrp_t_s`, `usrp_evm`, `usrp_ber` | (U,) | | USRP 프레임별 EVM과 BER |

**잔차 정의:** `psd_xy_m − p_tar`이며, 각 실행의 처음 `warmup_s`를 뺍니다. 50 Hz 프레임 기준이므로 PSD 잡음(약 5 µm)이 포함되고, 프레임 사이의 움직임과 FSM 지터는 일부만 보입니다. 따라서 분석 갈래의 0.5 ms 잔차와 바로 같지 않습니다. 모델과 비교할 때는 같은 형식으로 만든 리허설 결과(5절, PSD 프레임 기준)를 씁니다.

## 4. 실행

```bash
PY=/opt/anaconda3/envs/nlos/bin/python
PYTHONDONTWRITEBYTECODE=1 $PY experiments/run_empirical.py --session data/empirical/<session>
```

결과는 `results/empirical/<session>/figures/`에 있고, 그림 설명은 `FIGURES.md`에 있습니다.

## 5. 리허설(합성 로그)

실측 전에 파이프라인을 확인하기 위해, 분석 실행에서 저장한 제어기로 같은 형식의 로그를 만들 수 있습니다.

```bash
$PY experiments/make_synthetic_logs.py --analytical results/analytical/main --session data/empirical/synthetic_dryrun
$PY experiments/run_empirical.py --session data/empirical/synthetic_dryrun
```

합성 세션은 `session.json`에 `"synthetic": true`가 들어가고, 모든 E 그림에 "SYNTHETIC DRY RUN" 띠가 붙습니다. 논문이나 보고에 실측으로 쓰면 안 됩니다.

## 6. 해석할 때 주의

- **통신 지표는 HAPS 등가값입니다.** 벤치 PDA는 렌즈 없이 0.55 mm 빔을 받습니다. −3 dB가 되려면 327 µm 오프셋이 필요하므로, 수십 µm 잔차로는 전력이 거의 변하지 않습니다. USRP로 직접 잰 BER(E08)은 제어기와 무관하게 비슷할 것으로 예상합니다. 제어 성능은 E02–E07로 판단합니다.
- 진동 주입 시나리오에서 `command_rad`에 주입 신호를 섞어 기록하면, 제어기가 진동을 아는 것처럼 보이게 됩니다. 반드시 분리해서 기록합니다.
