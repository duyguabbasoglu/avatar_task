Türkçe Avatar Oluşturucu

LongCat-Video kullanarak Türkçe konuşan avatar üretimi

Kurulum

repoyu klonla
git clone --recursive https://github.com/duyguabbasoglu/avatar_task.git
cd avatar_task

gereksinimleri yukle
make setup

Kullanım

standart video üretimi
python run_avatar.py --text "Merhaba bu bir test videosudur"

ozellestirilmis video
python run_avatar.py --text "Yapay zeka gelişiyor" --image "assets/avatar/single/man.png" --prompt "A professional person speaking Turkish"

Test

sistemi test et
make test

Notlar

python 3.10 ve uzeri gerekli
cuda destegi onerilir
ffmpeg yuklu olmali
agirliklar otomatik indirilir
cikti dosyalari outputs klasorunde
