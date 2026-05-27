# Hokej trening

Repo vsebuje samo kodo, Docker konfiguracijo in majhne modele, ki so potrebni za zagon/nadaljevanje treninga. Lokalni logi in vecina checkpointov so ignorirani, da repo ostane uporaben za prenos na drugi racunalnik.

## Kloniranje na Linuxu

```bash
git clone https://github.com/Vulpes271/Hokej.git
cd Hokej
```

## Zagon z Docker Compose

Zgradi image in zazeni trening v ozadju:

```bash
docker compose up -d --build hockey-train
```

Spremljaj izpis treninga:

```bash
docker compose logs -f hockey-train
```

Ustavi trening:

```bash
docker compose stop hockey-train
```

Nadaljuj trening iz zadnjega checkpointa:

```bash
docker compose up -d hockey-train
```

Zacni trening na novo:

```bash
docker compose run --rm hockey-train python train_hockey.py --fresh
```

## TensorBoard

Zazeni TensorBoard:

```bash
docker compose --profile monitor up -d --build tensorboard
```

Odpri v brskalniku:

```text
http://SERVER_IP:6006
```

Ce si na istem racunalniku:

```text
http://localhost:6006
```

## Rocni Docker ukazi

Ce ne zelis uporabiti Compose:

```bash
docker build -t hokej-train .
docker run --rm -it \
  -v "$PWD/CircEnv/models:/app/CircEnv/models" \
  -v "$PWD/CircEnv/logs_hockey_active_goal:/app/CircEnv/logs_hockey_active_goal" \
  -v "$PWD/models:/app/models" \
  hokej-train
```

Trening privzeto pise checkpoint-e v `CircEnv/models/TQC_hockey_active_goal/` in TensorBoard loge v `CircEnv/logs_hockey_active_goal/`. Te datoteke ostanejo lokalno na Linux racunalniku, ampak se ne dodajajo v Git.
