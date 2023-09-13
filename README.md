# miraeasset-festa

2023 미래에셋 빅데이터 페스티벌을 준비하는 INHIVE팀의 저장소입니다.

저희 팀의 저장소는 크게 3 파트로 나뉘어있습니다. 현재 3 파트 모두 어느정도 완성이 되었고, 이를 서로 유기적으로 연결하는 과정을 결선까지 진행하고자 합니다.

- [miraeasset-festa](#miraeasset-festa)
  - [1. ML](#1-ml)
  - [2. DataEngineering](#2-dataengineering)
  - [3. Backend](#3-backend)
  - [서버 주소](#서버-주소)

해당 문서에서는 각 파트에 대한 간단한 설명과 코드 실행을 안내합니다. 

**글 말미에 보다 용이하게 실행하실 수 있도록 소스코드 환경이 온전히 구성되어 있는 서버 주소를 첨부합니다.**

----

## 1. ML
1. miraefest.yaml 에 따라 필요한 파이썬 패키지를 설치합니다.

```bash
cd conda env create -f miraefest.yaml
```

2. ML 폴더에 들어갑니다.

```bash
cd ML
```

3. cli.py 를 실행합니다.

```bash
python cli.py
```

4. 유의 사항


- 아나콘다 버전: 23.7.2

- cpu만 사용할 경우

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

- gpu 사용할 경우 
pytorch에 들어가서 본인 gpu cuda 버전에 맞는 torch 설치.


5. 가상 환경

```bash
pip install openai

pip install pandas

pip install numpy

pip matplotlib

pip tqdm

pip install scikit-learn

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## 2. DataEngineering

1. 루트 디렉토리에서 DataEngineering 폴더에 들어갑니다.

```bash
cd DataEngineering
```

2. requirements.txt 에 따라 필요한 파이썬 패키지를 실행해줍니다.

```bash
pip install -r requirements.txt
```

3. srcs 폴더에 들어갑니다.

```bash
cd srcs
```

4. 코드 실행

- 원래 코드의 경우, DB에 바로 접근하여 테이블과 상호작용을 합니다. 

- 해당 코드에서는 데이터 변조의 위험으로, 의도적으로 DB에 데이터를 쌓는 과정은 막아두었습니다.

```python
python api/naver/main.py

python api/dart/fss.py

python api/dart/main.py

python api/kis/handler.py
```

## 3. Backend

- 백엔드의 경우, Flask를 통해 서버를 띄우고 있습니다.

- 때문에 코드가 문제 없음을 보여드리기 위해, 서버 실행 중 `test_api.py`를 실행해주셔야 합니다.

- 때문에 총 2개의 터미널을 띄워주셔야 합니다. (백그라운드 프로세스로 돌리시면 서버가 멈춥니다.)

1. 루트 디렉토리에서 Backend 폴더에 들어갑니다.

```bash
cd Backend
```

2. requirements.txt 에 따라 필요한 파이썬 패키지를 실행해줍니다.

```bash
pip install -r requirements.txt
```

3. srcs 폴더에 들어갑니다.

```bash
cd srcs
```

4. app.py를 실행하여 서버를 구동시킵니다.

```bash
python app.py
```

5. 다른 쪽 터미널을 통해, **test_app.py**를 실행합니다.

```bash
python test_app.py
```

----

## 서버 주소

1. 터미널을 통해 다음의 명령어를 입력해줍니다.

```
ssh root@101.101.218.43
```

2. yes를 입력하고, 다음의 패스워드를 통해 서버에 접속합니다.

```
G2igr!$h8qch
```

3. festa-inhive 폴더에 들어갑니다.

```
cd /root/festa-inhive
```

4. 해당 폴더 내에서 위의 각 파트별 실행을 수행합니다.
