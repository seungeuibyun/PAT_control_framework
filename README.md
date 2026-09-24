# Receiver-FSM PAT control framework

`PAT (1).pdf`의 Sections II–IV, Tables I–II에 맞춘 수신기 측 PAT 시뮬레이션입니다.
기존 `config/`, `system_model/`, `solver/`, `simulation/`, `experiments/`, `results/` 구조를 유지합니다.
수치 결과 JSON은 덮어쓰지 않으며, `--replot`으로 기존 figure만 다시 만들 수 있습니다. 새 결과는 `model_version=receiver_fsm_pdf_v1`로 구분됩니다.

목적함수는 **통신 검출기에서 수집한 광전력 `P_D,k(theta)` [W]**입니다.
PSD centroid는 명령 초기화와 진단에 사용합니다. 송신기 조향, 이동하는 목표점 추적,
centroid 오차 최소화, SMF coupling 목적함수는 이 버전의 실험 대상이 아닙니다.

## 실행

NumPy, SciPy, Matplotlib, Pillow가 필요하고 기본 SSFM 자동미분 비교에는 PyTorch도 필요합니다.

```bash
python -m pip install -r requirements.txt

# 물리 파라미터는 PDF 값, 계산 해상도는 빠른 확인용
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/compare_pat.py \
  --preset demo --name my_pdf_power_demo --intervals 20 --gif

# 기본 preset은 paper: SSFM 1024², M=1024, Nc=128²; GIF도 기본 생성
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/compare_pat.py \
  --device mps --name my_pdf_power_reference --intervals 20

# Table II: 100개 channel seed × 5개 feature seed의 교차 실험 (계산량 큼)
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/compare_pat.py \
  --paper-ensemble --name my_pdf_ensemble --intervals 20

# 채널 반복만 먼저 확인
python experiments/compare_pat.py --preset demo --name my_pdf_repeats \
  --repeat 3 --feature-seeds 1 --intervals 10

# 추가 실험: 고정 검출기에 대한 주기적인 송신 각도 오차
python experiments/compare_pat.py --preset demo --name my_pdf_periodic \
  --disturbance periodic --target-period 0.5 --intervals 50 --gif

# Torch 없이 실행 가능한 비교
python experiments/compare_pat.py --name my_pdf_cpu --frozen-backend scipy \
  --solvers frozen_pinn pid linear_mpc ssfm_oracle_cpu no_control
```

`--preset`을 생략하면 `paper`를 사용합니다. 빠른 확인용 256² 격자는 `--preset demo`로 명시합니다.
GIF는 기본 생성되며 `--no-gif`로 끕니다. `--preset paper`만으로 500개 반복이 시작되지는 않습니다. 반복 수는 `--repeat`,
`--feature-seeds` 또는 `--paper-ensemble`로 선택합니다. `--intervals`는 각 실행의
시간 구간 수이며 채널 seed 반복 수와 다릅니다. 기존 `results.json`이 있으면
덮어쓰기를 거부하므로 새로운 `--name`을 사용하세요.

Torch receiver는 CPU/CUDA/MPS를 지원합니다. Apple Silicon에서는 다음처럼 실행합니다.

```bash
python experiments/compare_pat.py --preset demo --device mps \
  --name my_pdf_mps --intervals 20 --gif
```

기본 `--frozen-backend auto`는 Frozen-PINN과 SSFM oracle의 수신 광학계 질의에
`--device`를 적용합니다. `--device auto`는 CUDA, MPS, CPU 순으로 선택합니다.
MPS에서는 복소장을 실수부·허수부의 두 float32 tensor로 표현하고 실수 행렬곱과
자동미분으로 전력·미분값을 계산합니다. `--dtype float64`를 주어도 MPS 계산은
float32이며 JSON의 `requested_dtype`, `effective_dtype`, `receiver_complex_backend`에 기록됩니다.
명시적으로 요청한 MPS를 사용할 수 없으면 CPU로 조용히 바꾸지 않고 오류를 냅니다.

MPS/CUDA에서는 Frozen-PINN의 층별 Galerkin 행렬 `Pᵀ diag(δn) P`를 GPU에서 한 번
구성하고, 같은 행렬로 CPU float64 RK45를 적분합니다. 이전처럼 RHS 평가마다
큰 기저 변환과 DST를 반복하지 않습니다. 기저·collocation·RK45 tolerance는 유지합니다.
수신 광학계 질의도 GPU를 사용합니다. SSFM 대기 전파는 CPU입니다. 따라서 이 비교의
속도는 구현된 혼합 backend의 결과이며, 전체 GPU SSFM 대비 우위로 해석하면 안 됩니다.
JSON에 대기 backend, 행렬 구성, ODE 적분 시간, offline operator 구성 시간이 기록됩니다.
`--frozen-operator matrix_free`는 이전 CPU RHS로 비교할 때 사용합니다.
`--frozen-backend scipy`를 명시하면 Frozen-PINN 수신 질의도 CPU 경로를 사용합니다.

## 시스템 모델과 문서 대응

| 코드 | 구현 내용 | PDF |
|---|---|---|
| `config/settings.py` | SI 단위 물리·수치 설정, paper/demo preset, 설정 검증 | Tables I–II |
| `system_model/optical_system.py` | Gaussian 송신장, 알려진 Tx 각도 오차, 대기 SSFM | (1)–(4) |
| 같은 파일의 `reduce_field`, `fsm_field` | 원형 입구 개구, 면적 보정 `sqrt(eta_G)/abs(m_G)`, 수신 FSM 위상 | (5)–(7) |
| 같은 파일의 detector/PSD 연산 | 전파·렌즈·beam splitter·두 검출면 | (8)–(20), (38)–(42) |
| `system_model/turbulence.py` | 고도 `h=d-z`의 HV profile, modified von Karman spectrum | Section IV |
| `solver/frozen_pinn.py` | 고정 tanh 특징, 경계 변환, SVD, 초기 LS, RK45 | (21)–(37) |
| `solver/optimization.py` | PSD 초기 명령, 축별 projection, normalized gradient ascent, 최종 양자화 | (43)–(47) |
| `solver/baselines.py` | PSD PID, one-step linear MPC, finite-difference SSFM oracle | 비교 실험 |
| `solver/differentiable_ssfm.py` | 별도 SSFM 예측 + receiver-only 자동미분 oracle | 비교 실험 |

물리 기본값은 1550 nm, 20 km, 10 mW, `w0=50 mm`, Tx 각도 표준편차 5 µrad,
경로 손실 1 dB, 입구 직경 100 mm, reducer `mG=0.1`, `etaG=0.97`,
렌즈 초점거리 200 mm, `l1=l2=50 mm`, `lP=lD=150 mm`, splitter `T=0.62`, `R=0.27`,
PSD 10×10 mm², 통신 검출기 반경 75 µm입니다.
FSM은 `CF=2I`, 범위 ±1 mrad/축, slew 10 mrad/s/축, `Tc=10 ms`,
양자화 1 µrad, PSD 위치 잡음 표준편차 5 µm/축입니다.

대기장의 계산 좌표는 ±1 m이고, 통신 검출면의 적분 간격은 별도로 **2 µm**입니다.
뒤 초점면의 광학 연산은 `P_(l2+lD) L_f P_l1`의 Collins 적분을 사용합니다.
`A=0, B=f, D=1-l1/f`인 scaled Fourier transform을 직접 계산하여 렌즈의 급격한
위상과 작은 검출기를 대기 격자에서 잘못 샘플링하는 문제를 피합니다.
PSD는 별도 FFT 격자에서 유한한 10×10 mm² 활성 영역에 대해 적분합니다.
전체 PSD를 Fourier 주기 복제 없이 덮을 수 없는 receiver grid 설정은 거부합니다.
현재 광학 연산은 PDF 실험의 뒤 초점면 조건을 검증하며 임의의 relay optics는 지원하지 않습니다.

## Frozen-PINN 구현 선택과 정확도

공간 특징 `tanh(w_m·r/H+b_m)`는 한 번 샘플링한 뒤 고정합니다. 문서에 구체적인
구성이 주어지지 않은 `A_boundary`는 **고정 Gaussian window + 유한 Dirichlet sine
projection**으로 구현했습니다. 이는 선형 변환이며 모든 경계점에서 zero-field
조건을 만족하고 analytic Laplacian을 제공합니다. 이는 명시적으로 선택한 수치 구현이며,
PDF가 같은 변환을 사용했다고 가정하지 않습니다.

변환된 특징의 SVD에 `1e-6 sigma_1` cutoff를 적용합니다. 역특이값으로 계수 단위를
재조정하여 conditioning을 개선하지만 공간 함수의 span은 바뀌지 않습니다.
전파 RHS는 transformed collocation basis의 least-squares 식 `(35)`를 평가합니다.
CPU 기본 경로는 matrix-free이고 MPS/CUDA 기본 경로는 같은 연산자의 층별 행렬을 재사용합니다.
학습 데이터, 역전파 학습, atmospheric field를 PSD로 추정하는 단계는 없습니다.

각 구간에서 알려진 송신장·대기 입력을 사용해 `c(0)`을 새로 구하고 `dc/dz=G(z)c`를
RK45로 적분합니다. 층 경계의 불연속점을 건너뛰지 않도록 적분을 구간별로 이어갑니다.
전체 경로 계산은 PAT 구간당 한 번이며 trial FSM 명령마다 반복하지 않습니다.
`dU_rx/dtheta=0`이고, 미분은 수신 FSM 이후의 광학 연산에만 적용합니다.
Torch 옵션도 동일한 RK45 atmospheric basis를 사용합니다.

**실행 성공은 PINN의 수렴 또는 논문 성능 재현을 의미하지 않습니다.** 이 링크에서
좁은 송신 Gaussian과 큰 Tx 각도 오차는 기저 표현에 부담을 주며, 초기장 오차와 전파
오차가 서로 다르게 변할 수 있습니다. 따라서 대조군 순위나 속도 우위를 강제하지 않습니다.
JSON에는 아래 값이 실제 SSFM 평가와 함께 저장됩니다.

- `initial_field_relative_error`: fitting grid와 다른 격자에서 측정한 초기장 오차
- `receiver_field_relative_error`, `receiver_intensity_relative_error`: reducer 출력에서 PINN/SSFM 차이
- `power_prediction_relative_error`: 실제 적용된 양자화 명령에서 전력 예측 오차
- `coefficient_power_ratio`, `expected_extinction_ratio`: 수치 ODE의 전력 소멸 확인
- `basis_rank`, `ode_nfev`, `atmosphere_integrations_this_interval`: 실제 계산 규모

`--spectral-side`, `--hidden-width`, `--collocation-side`, feature seed, SSFM grid,
layer thickness 및 mode 수를 바꿔 **관심 있는 검출 전력과 최종 명령의 수렴**을 확인해야 합니다.
더 큰 bandwidth만으로 정확도가 개선된다고 가정하지 마세요.

| 수치 설정 | paper | demo |
|---|---:|---:|
| SSFM 격자 | 1024² | 256² |
| SSFM nominal step / 대기 layer | 50 m / 50 m | 500 m / 500 m |
| Frozen tanh 특징 M | 1024 | 1024 |
| Collocation Nc | 128² | 96² |
| RK45 rtol / atol | 1e-6 / 1e-8 | 동일 |
| 통신 검출면 sampling | 2 µm | 동일 |
| Receiver pupil grid | 1024² | 512² |
| Random spectral modes / layer | 128 | 32 |

**두 설정의 1024는 단위가 다릅니다.** PINN의 `M=1024`는 공간 특징 수이고,
SSFM의 `N=1024`는 축당 격자 수이므로 총 1,048,576개 셀입니다. PINN의 계산 점 수는
별도의 `Nc=128²`입니다. 숫자를 같게 맞추는 것만으로 정확도가 같아지지는 않습니다.
PDF 기준 비교는 paper preset을 사용하고, demo 결과를 paper 해상도의 SSFM과 혼동하지 마세요.

추가 수치 선택인 boundary bandwidth(48²), window 폭(0.35H), random feature 분포,
receiver pupil sampling 및 atmospheric mode 수는 설정에 기록됩니다.

## 대기·센싱·비교 실험의 범위

HV profile은 `C0=1.7e-14`, `vHV=21 m/s`, outer scale 10 m, inner scale 5 mm를 사용합니다.
Modified von Karman phase PSD를 log-frequency random Fourier quadrature로 근사하고,
층별 적분 세기를 `r0^(-5/3)=0.423 k0² integral(Cn² dz)`에 맞춥니다.
`delta_n=phase/(k0 dz)`인 piecewise-constant Markov layer model을 두 propagator가 공유합니다.
이는 유한 mode 수를 가진 합성 채널이며 실제 측정 대기장 또는 완전한 3D turbulence는 아닙니다.

HV 및 phase PSD 수식 확인 자료:
[HV profile의 연구 문헌](https://www.mdpi.com/2073-4433/13/2/162),
[AOtools phase-screen 구현](https://aotools.readthedocs.io/en/v1.0.1/_modules/aotools/turbulence/phasescreen.html).
코드는 이 수식을 이용한 별도의 random-mode quadrature를 구현합니다.

문서에서 지정하지 않은 시간 통계는 기본적으로 PAT 구간 사이 독립 realization과
독립 Gaussian Tx 각도 오차로 선택했습니다. `periodic`은 추가 진단 실험이며
`zero`는 Tx 각도 오차를 제거합니다. 전송 modulation은 구간 평균 전력에만 반영합니다.

PSD reference와 Jacobian은 정렬된 뒤 초점면의 이상적 calibration
(`p_tar=detector_center`, `J_F=f CF`)을 사용합니다. 하드웨어의 통신 전력 sweep으로
측정한 calibration을 대신했다고 주장하지 않습니다. `--detector-offset X Y`로
검출기 정렬 오차를 지정할 수 있습니다. 이 구현은 실제 실험 장비와 연결하지 않습니다.
PSD power 잡음은 표에 값이 없으므로 0이고, 유효하지 않은 PSD 위치는 초기화에 사용하지 않습니다.

모든 제어기는 해당 구간의 **이전 명령**에서 PSD를 읽습니다. 각 제어기의 명령은 다를 수 있지만
같은 구간의 잡음 표본과 채널은 재현됩니다. PID/MPC는 PSD만 사용하며 PINN/SSFM oracle은
알려진 대기를 추가로 사용하는 model-based 비교입니다. PID/MPC 이득은 비교용 설정입니다.

실행시간에는 각 model-based 제어기 자체의 대기 예측, 수신 광학 질의 및 명령 계산이 포함됩니다.
시뮬레이터가 만드는 PSD 측정, 독립 truth evaluation, plot 생성 및 offline basis SVD/공간 기저 구성은 제외합니다.
센서 기반 제어기와 모델 기반 제어기의 런타임 범위를 동일한 연산량으로 해석하면 안 됩니다.

## 저장되는 figure와 JSON

기존 figure 이름과 전반적인 비교 형식을 유지합니다.

- `objective_comparison.png`: 실제 수신 전력 [mW] + 무제어 대비 전력 변화율의 누적분포
- `objective_comparison_all.png`: No control을 포함한 전력 비교
- `power_gain_comparison.png`: 같은 채널의 No control 대비 전력 변화율 [%]; 겹치는 곡선의 차이 확인
- `runtime_comparison.png`: 구간당 실행시간(log 축) + 대기 예측/수신 질의/기타 연산 비율
- `centroid_trajectories.png`: 제어기별 **PSD** XY 산점도와 고정 calibration reference; 보조 진단
- `receiver_xy_comparison.png`: 같은 마지막 구간의 모든 제어기에 대한 **통신 검출면** 광강도; 흰 원은 검출기 활성 영역
- `comparison.gif`: 통신 검출면 XY 광강도(시간 전체에 고정된 log 색상 범위), 절대 전력, 무제어 대비 변화율
- `output_manifest.json`: 실제 생성된 figure/GIF 목록과 프레임 수
- `results.json`: 설정, 실제 전력 [W], 이전/적용 FSM 명령, PSD 측정, 예측 전력, runtime, 근사 오차

수치 실험을 다시 돌리지 않고, 저장된 채널/명령으로 그림과 GIF를 다시 만드는 명령:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/compare_pat.py \
  --replot results/pdf_power_demo
```

이 명령은 제어기를 다시 최적화하지 않으며 `results.json`과 기존 runtime을 보존합니다.
GIF의 XY 광학장만 저장된 seed와 명령으로 재계산합니다. 새 실험은 이미 계산한 truth장을
재사용하므로 GIF 때문에 대기 전파를 다시 돌리지 않습니다. 이전 centroid 모델 JSON은
현재 모델로 재해석하지 않도록 거부합니다.

여러 번 반복하면 `run_000/`, `run_001/`, …와 aggregate JSON/figure가 만들어집니다.
Feature seeds는 같은 channel seeds를 공유하는 교차 실험입니다. Aggregate band는
서술적 표준편차이며 500개의 독립 channel 표본에 대한 신뢰구간이 아닙니다.
이전 버전의 centroid 결과와 새 버전의 power 결과를 직접 합산하지 마세요.

## 검증과 runtime 실험

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/validate_framework.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/validate_framework.py --device mps
python experiments/quick_smoke_test.py
python experiments/runtime_scaling.py --grid-sizes 128 256 512 --queries 1 5 20
```

검증은 analytic Gaussian diffraction, extinction, aperture/reducer normalization,
FSM 반사 이득, analytic/FD/autograd 전력 gradient, 축별 제약과 양자화,
invalid PSD 처리, frozen boundary 및 ODE 재사용, seed 재현성을 확인합니다.
Runtime benchmark는 **대기 예측 한 번**과 **cached receiver query**를 분리해 기록합니다.

## 이번 변경에서 실제 실행한 결과

검증 환경은 `/opt/anaconda3/envs/ml/bin/python`과 CPU입니다. 2026-09-22에 아래 실행을
완료하고 PNG 및 20-frame GIF를 확인했습니다. 기본 물리 설정을 바꿔 제어기 간 차이를
확대하거나 특정 제어기가 이기도록 결과를 수정하지 않았습니다.

| 실행 | 저장 위치 | Frozen-PINN 전력 예측 오차 |
|---|---|---|
| demo, 20구간, 5개 제어기 | `results/pdf_power_comparison/` | 평균 2.05%, 최대 5.99% |
| paper 해상도, 1구간, PINN/SSFM/무제어 | `results/pdf_paper_single_interval/` | 해당 구간 27.51% |

Demo 평균 실제 수신 전력은 PINN 0.269263 mW, SSFM oracle 0.269308 mW였습니다.
Demo 구간당 평균 runtime은 각각 2.187 s, 0.320 s였으므로 이 실행에서 PINN 속도 우위는
확인되지 않았습니다. Paper 단일 구간의 실제 수신 전력은 PINN 0.031921 mW,
SSFM oracle 0.032262 mW입니다. **이 구간에서 PINN의 예측 전력 오차가 크므로,
이 실행을 논문 성능 재현 또는 PINN 수렴 검증으로 사용하면 안 됩니다.**
100×5 전체 ensemble은 실행하지 않았습니다.

추가로 physical regression suite, Torch Frozen wrapper, FD oracle,
2×2 channel/feature seed 교차 반복 및 aggregate 출력, 기존 결과 덮어쓰기 방지,
최소 runtime-scaling 실행을 확인했습니다.

MPS 지원 복구 후 실제 Mac MPS에서 native CPU/실수부·허수부 구현의 전력 및 gradient
일치, float32 tensor의 GPU 배치, Frozen-PINN과 SSFM oracle의 2구간 제어 실행을
검증했습니다. `experiments/validate_framework.py --device mps`로 같은 검증을 실행할 수 있습니다.


## 속도·출력 수정 확인 (2026-09-22)

`results/pdf_power_demo/`의 기존 100구간 수치 결과는 보존하고 figure와 100-frame GIF를
다시 생성했습니다. 당시 GIF가 없던 원인은 CLI가 `--gif`를 주어야만 저장하는 설정이었기
때문입니다. 현재는 기본 생성, `--no-gif`로 비활성화합니다.

이 100구간의 무제어 PSD 위치 오차는 평균 7.34 µm이고 검출기 반경은 75 µm입니다.
채널별 전력 변동에 비해 수신 steering으로 바뀌는 전력은 작습니다. 입구 개구 전의 손실은
수신 FSM으로 회복되지 않습니다. PID/MPC의 PSD 잡음(5 µm/축) 역시 잔여 정렬 오차에
비해 작지 않습니다. 이 조건에서는 모델 기반 제어의 평균 이득도 작게 나올 수 있습니다.
절대 전력, 무제어 대비 변화율, 분포를 분리하여 표시하고 물리 설정은 바꾸지 않았습니다.

느렸던 주원인은 유효 rank 약 710–727의 큰 기저에 대해 RK45의 RHS 평가마다 DST와
기저 변환을 반복한 것이었습니다. 이제 GPU에서 층별 potential을 한 번 조립합니다.
Paper 설정의 **동일 기저·채널·RK45 tolerance** 2구간 비교에서는 대기 준비시간이
평균 12.477 s → 4.675 s (2.67배)로 줄었고, 기존 RK45 대비 전력 차이는 최대
0.00348%, gradient 상대 차이는 최대 0.01093%였습니다.
`results/pinn_operator_paper_mps.json`에 개별 시간과 offline 구성 비용이 기록됩니다.
이 수치는 전체 제어시간이 아닌 대기 예측과 receiver field upload 시간입니다.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/benchmark_pinn_operator.py \
  --device mps --preset paper --intervals 2 --output results/my_operator_benchmark.json
```

CPU 및 실제 MPS에서 행렬 없는 RHS와 층별 행렬 RHS의 일치, RK45 궤적, 전력/gradient,
장치 선택, 2구간 제어기 실행을 검증했습니다. 이 최적화는 원래 PINN 공간 기저의
SSFM 대비 근사 오차를 해결하지 않습니다. 기존 demo 100구간 전력 예측 오차는 평균
4.06%, 최대 73.83%였으므로 작은 실제 제어 성능 차이를 정확한 PINN 전파의 증거로
해석하면 안 됩니다. 10 ms 제어주기 충족도 아직 아닙니다.


기본 preset/기본 GIF 동작을 함께 확인한 `results/pdf_paper_mps_verified/`에는
SSFM 1024², PINN M=1024/Nc=128², 5개 제어기의 3구간 결과와 GIF가 있습니다.
전체 온라인 제어시간은 평균 PINN 4.894 s, SSFM oracle 18.491 s였습니다.
PINN은 MPS 층별 행렬 구성 + CPU RK45 + MPS receiver, SSFM은 CPU 대기 전파 +
MPS receiver 조합입니다. 이 작은 표본의 PINN 전력 예측 오차는 평균 9.37%, 최대
27.51%이므로, 속도 개선과 독립적으로 공간 근사의 정확도 검토가 필요합니다.
