.PHONY : train
train:
	CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 train.py
.PHONY : sample
sample:
	python sample.py --w=2.0
.PHONE : clean
clean:
	rm -rf __pycache__
	rm -rf model/*
	rm -rf sample/*
