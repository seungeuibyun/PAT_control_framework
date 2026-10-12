# 2026-10-12 테스트베드 인계

## 광학 구성

현재 송신단은 Zaber X-LSQ450A-E01C 위의 F230APC-1550 collimator와 GBE10-C 10×
expander입니다. 두 소자는 함께 평행 이동하며 housing 간격은 약 5 mm입니다.
연구실 송수신 거리는 약 1 m로 계획했으며 정밀 측정값은 없습니다. 파장은 1562 nm입니다.
수신 GBE10-C는 큰 면으로 받고 작은 면을 FSM 쪽으로 향하게 한 10× reducer입니다.
이후 MR-15-30 FSM → beam splitter → CONEX-PSD10GE와 PDA015C2의 두 branch입니다.
PD branch에는 별도 집광 렌즈가 없고 PD 전원은 외부 전원입니다. PD 출력은 RIGOL CH1입니다.

이전에 PINN을 확인한 구성은 **고정 breadboard source → FSM → BS → PSD/PD**였습니다.
`evidence/fixed_source_20261012/`의 모델과 폐루프 기록은 이 구성의 결과입니다.
GBE 두 개와 stage-mounted 송신단을 설치한 후 같은 모델을 다시 검증한 결과는 없습니다.

## 가장 최근에 확인한 상태

| 장비 | 연결·상태 |
|---|---|
| FSM | COM5, MR-E-3/MR-15-30. 최종 manual current mode, 양 축 명령 0 mA; 자동 제어 정지 |
| PSD | COM4, 921600 baud. 최근 값 1%, Y가 +5 mm boundary에 닿아 유효 위치 아님 |
| Source | COM3, Termite 9600/8N1, Optilab DFB-SM-M-1560-25. 최근 34 mA, 장치 보고 3.35 dBm |
| RIGOL | 192.168.10.2, PC의 해당 LAN 192.168.10.3, CH1/DC/1×, 최근 0.1 V/div |
| PDA015C2 | 전원 ON으로 확인 후 비교. CH1은 dark 부근, 충분한 수광 확인되지 않음 |
| Zaber | COM6, 115200 baud, ASCII/address 1. 최근 `WR=NoReferencePosition`, absolute position 유효하지 않음 |
| GUI | http://192.168.0.30:8765/ ; 모니터링 PC 192.168.0.36 |

0 mA는 거울 자세가 이전 위치와 정확히 같다는 뜻이 아닙니다. 실제 자세는 feedback으로 확인합니다.
최근 Zaber native position `4550585`는 `limit.max`와 같았으며 `WR`가 있어서 물리적 위치로
인증할 수 없습니다. 경고 원인은 확정하지 않았습니다. 위치를 임의로 재설정하거나 home을 실행하지 않았습니다.
`testbed_config.example.json`은 연결값 참고용이며 server 설정은 CLI로 전달합니다.

## 새 GBE 배치에서 수행한 진단

1. Zaber ±0.5 mm, 이후 PD 전원 ON 상태에서 ±2 mm의 bounded relative 이동을 actual
   readback으로 확인했습니다. 유효한 수광 증가를 얻지 못했고 당시 origin으로 복귀했습니다.
2. 다음 +3/+5 mm 탐색 시도는 `WR` 경고로 probe 이동 전에 중단했습니다.
3. FSM의 2축 bounded raster 35개 자세를 실제 feedback과 함께 측정했습니다.
   PSD interior spot을 찾지 못했으며 source/stage 이동 없이 FSM current 명령을 0으로 반환했습니다.
4. PD 전원 ON, source current 0에서 dark 평균 **33.976592 mV**를 측정했습니다.
   source 32/34 mA에서 CH1 평균은 각각 **34.188969/34.228404 mV**였습니다.
   약 **0.039435 mV** 차이는 관측된 표준편차 약 **0.039–0.048 mV**와 비슷한 크기여서,
   충분한 수광 회복을 확인하는 근거로 삼지 않았습니다.
5. 사용자가 IR 카드로 확인한 빔은 FSM의 보는 방향 기준 **왼쪽 가장자리**를 지나고 있었습니다.
   FSM tilt는 반사 방향을 바꾸므로 거울 밖으로 지나가는 incident footprint의 평행 위치를
   대신 정렬해 주지 못합니다. 수신 GBE/FSM 입사 정렬을 먼저 확보해야 합니다.

이 진단은 PINN 운전이 아니며 `PINN_used=false`, `new_geometry_calibrated=false`로 남겼습니다.
이후 PINN을 활성화하거나 장비를 움직이는 동작은 이번 GitHub publication에서 수행하지 않았습니다.

## 후속 실험 순서

1. 송신단을 고정하고 수신 GBE 출구 빔이 FSM 유효 거울면에 들어오는지 확인합니다.
   stage 운동을 사용할 때는 beam footprint가 수신 aperture 안에 유지되는 범위를 별도로 확인합니다.
2. PSD native spot이 신호/edge/포화 guard를 통과하도록 branch를 정렬합니다.
   FSM과 PSD의 좌표·단위는 서로 다르므로 `+Y`의 표시 부호가 같다는 조건으로 판단하지 않습니다.
3. source와 PD 전원을 고정하고 dark/PSD electrical offset을 새로 측정합니다.
   dark calibration은 source current를 0으로 바꾼 뒤 복구하므로 laser session owner를 확인합니다.
4. FSM 양 축을 probe하고 실제 feedback과 native PSD 좌표의 25점 map을 새로 수집합니다.
   학습과 withheld/derivative 검사를 통과한 새 모델을 사용합니다. 기존 함수의 source 전류
   check는 32 mA로 되어 있으므로 조건을 맞추거나 새 실험의 검증된 조건으로 일관되게 수정해야 합니다.
5. 먼저 PSD 중심 유지부터 확인합니다. 이후 같은 FSM 자세에서 PD를 수동 정렬하여 PSD 중심과
   최대 PD 수광점의 coincidence를 실제 2축 전력 측정으로 확인합니다.
6. Zaber reference와 이동 한계를 확인한 후 independent source disturbance 실험을 추가합니다.
   stage position은 evaluation용 ground truth로만 기록하고 receiver controller 입력으로 주지 않습니다.
7. 20 km/HAPS 대기 model이나 PD 전력 목적함수와 결합하려면 별도의 receiver observation model,
   광학 파라미터 및 power calibration 검증이 필요합니다. 현재 로컬 derivative constraint가
   20 km channel을 추정했다는 결론으로 확장하면 안 됩니다.

## 코드 유지 시 확인할 것

`calibration/manual_fsm.py`는 temporary XY servo gains/current limits를 설정하고 실제
firmware readback을 확인합니다. 내부 servo와 외부 PINN을 구별하고 mode 변경 후 retained
integral 및 output state 복구 절차를 유지해야 합니다. 새 장비·firmware에서 동작을 다시 확인하세요.
`calibration/calibrate_fsm_psd.py`는 별도의 직접 current probe 도구입니다. broker가 serial을
소유하는 동안 별도 direct serial owner를 실행하는 경로로 사용하지 않습니다.

`measure_mounted_source_power.py`는 당시 승인된 32 → 34 mA 비교를 기록한 도구이며, 일반적인
자동 출력 증가 정책이 아닙니다. 호출하면 source current와 PSD electrical offsets를 실제로
변경합니다. `inspect_mounted_receiver.py`도 실행하면 선택된 stage/FSM을 움직입니다.
먼저 `check_psd_position_pinn.py`와 `reproduce_validation.py`로 장비 없는 검증을 수행하세요.
