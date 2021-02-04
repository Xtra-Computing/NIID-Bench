for partition in noniid-labeldir noniid-#label1 noniid-#label2 noniid-#label3 iid-diff-quantity homo
do
	for alg in fedavg scaffold fednova
	do
		python experiments.py --model=simple-cnn \
			--dataset=cifar10 \
			--alg=$alg \
			--lr=0.01 \
			--batch-size=64 \
			--epochs=10 \
			--n_parties=100 \
			--rho=0.9 \
			--comm_round=500 \
			--partition=$partition \
			--beta=0.5\
			--device='cuda:0'\
			--datadir='./data/' \
			--logdir='./logs/' \
			--noise=0\
			--sample=0.1\
			--init_seed=0
	done

	for mu in 0.001 0.01 0.1 1
	do
		python experiments.py --model=simple-cnn \
			--dataset=cifar10 \
			--alg=fedprox \
			--lr=0.01 \
			--batch-size=64 \
			--epochs=10 \
			--n_parties=100 \
			--rho=0.9 \
			--mu=$mu
			--comm_round=500 \
			--partition=$partition \
			--beta=0.5\
			--device='cuda:0'\
			--datadir='./data/' \
			--logdir='./logs/' \
			--noise=0\
			--sample=0.1\
			--init_seed=0
	done
done
