for model in vgg resnet
do
	for partition in noniid-labeldir iid-diff-quantity
	do
		for alg in fedavg fedprox scaffold fednova
		do
			python experiments.py --model=$model \
				--dataset=cifar10 \
				--alg=$alg \
				--lr=0.01 \
				--batch-size=64 \
				--epochs=10 \
				--n_parties=10 \
				--rho=0.9 \
				--mu=0.01 \
				--comm_round=100 \
				--partition=$partition \
				--beta=0.1\
				--device='cuda:0'\
				--datadir='./data/' \
				--logdir='./logs/' \
				--noise=0\
				--init_seed=0
		done
	done
	
	for alg in fedavg fedprox scaffold fednova
		do
			python experiments.py --model=$model \
				--dataset=cifar10 \
				--alg=$alg \
				--lr=0.01 \
				--batch-size=64 \
				--epochs=10 \
				--n_parties=10 \
				--rho=0.9 \
				--mu=0.01 \
				--comm_round=100 \
				--partition=homo \
				--beta=0.1\
				--device='cuda:0'\
				--datadir='./data/' \
				--logdir='./logs/' \
				--noise=0.1\
				--init_seed=0
		done
done
