# Inštalacija okolja CircEnv

## Novo Python okolje

Naslednji ukazi so nujni, le če nimate ustreznega Python okolja. Sicer aktivirajte raje `rl_test` okolje in pojdite na poglavje [Instalacija okolja in paketov](#instalacija-okolja-in-paketov).

```console
conda create --name circ_env python=3.8
conda activate circ_env
```

## Inštalacija `swig` paketa

Preverite, če je inštaliran `swig` z ukazom

```console
swig -version
```

Ukaz bi moral vrniti nekaj podobnega spodnjemu izpisu

```console
SWIG Version 3.0.12

Compiled with g++ [x86_64-pc-linux-gnu]

Configured options: +pcre

Please see http://www.swig.org for reporting bugs and further information
```

Če niste dobili podobnega izpisa, oziroma ste dobili obvestilo

```console
Command 'swig' not found, but can be installed with:

sudo apt install swig
```

je potrebno `swig` inštalirati.

```console
sudo apt update
sudo apt -y install swig
```

Geslo za sudo je `3krat4`.

## Instalacija okolja in paketov

Ime mape za okolje mora biti `CircEnv`. Pomaknite se v mapo, ki vsebuje mapo `CircEnv`.

Primer:

```console
ll
total 24
drwxrwxr-x  4 student student 12288 May  3 09:51 ./
drwxrwxr-x 16 student student  4096 Apr 13 16:01 ../
drwxrwxr-x  4 student student  4096 May  3 09:59 CircEnv/
drwxrwxr-x  4 student student  4096 May  3 09:50 SimpleEnvs/
```

Nato naredite inštalacijo okolja in paketov z ukazi:

```console
pip install -e CircEnv

pip install box2d box2d-kengz
```
