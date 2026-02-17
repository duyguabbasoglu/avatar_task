.PHONY: setup test clean run

setup:
	@echo "paketler yukleniyor"
	pip install -r requirements.txt
	@echo "alt moduller guncelleniyor"
	git submodule update --init --recursive
	@echo "model dosyalari indiriliyor"
	huggingface-cli download meituan-longcat/LongCat-Video-Avatar --local-dir ./weights/LongCat-Video-Avatar

test:
	@echo "testler baslatiliyor"
	python3 -m unittest discover tests

clean:
	@echo "gecici dosyalar siliniyor"
	rm -rf outputs/*.mp4
	rm -rf __pycache__
	rm -rf tests/__pycache__
	rm -f *.mp3
	rm -f assets/avatar/*.json

run:
	@read -p "metin girisi: " text; \
	python3 run_avatar.py --text "$$text"
