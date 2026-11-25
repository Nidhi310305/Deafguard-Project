\# DeafGuard



DeafGuard is a multimodal assistive prototype for Deaf and hard-of-hearing users. It combines environmental sound awareness with selected Indian Sign Language recognition.



\## Main features



\- Seven-class environmental sound classification

\- YAMNet-based audio feature extraction

\- Fifteen-sign isolated ISL recognition

\- MediaPipe hand landmarks

\- StandardScaler and RBF-kernel SVC classifier

\- Confidence and temporal filtering

\- Laptop HTTP/JSON alert server

\- Android visual, speech, notification, and vibration response

\- Confirmation-gated HELP/SOS prototype



\## ISL vocabulary



\- doctor

\- emergency

\- fall

\- food

\- hello

\- help

\- no

\- pain

\- safe

\- stop

\- thank\_you

\- tired

\- unsafe

\- water

\- yes



\## Sound vocabulary



\- baby\_crying

\- door\_knocking

\- emergency\_siren

\- footsteps

\- glass\_breaking

\- other\_unknown

\- vehicle\_horn



\## Current architecture



The laptop performs Python-based inference. The Android companion application receives validated JSON alerts over a local Wi-Fi network.



```text

Microphone/Webcam

&#x20;       ↓

Laptop inference gateway

&#x20;       ↓

Sound branch + ISL branch

&#x20;       ↓

Confidence and temporal decision layer

&#x20;       ↓

HTTP/JSON alert server

&#x20;       ↓

Android companion application

&#x20;       ↓

Visual output, speech, vibration, notification, SOS confirmation

Results

ISL dataset: 627 samples

ISL classes: 15

ISL train/test split: 501/126

Recorded preliminary ISL holdout accuracy: 96.0%

Sound classes: 7

Recorded sound test accuracy: approximately 92.9%

These are prototype evaluation results and should not be interpreted as universal real-world performance.

Running the project

Activate the virtual environment:

bat

.venv-yamnet\\Scripts\\activate

Start the laptop server:

bat

python deafguard\_demo\_launcher.py

Start the ISL recognizer in another terminal:

bat

python isl\_major\_live\_demo.py

Start the Streamlit interface:

bat

streamlit run deafguard\_web\_app.py

Limitations

The current system recognizes isolated signs rather than continuous ISL sentences.

The laptop is the current inference gateway.

Android-only inference is future work.

SOS does not currently send real SMS messages or contact emergency services.

The sound classifier recognizes only the selected seven sound classes.

