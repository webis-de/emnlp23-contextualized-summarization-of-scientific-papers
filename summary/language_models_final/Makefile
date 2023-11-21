VERSION=1.0.0

guard-%:
	@ if [ "${${*}}" = "" ]; then \
		echo "Environment variable $* not set"; \
		exit 1; \
	fi

build: guard-DOCKER_USER
	docker build --tag ${DOCKER_USER}/language-models:${VERSION} .

push: guard-DOCKER_USER
	docker push ${DOCKER_USER}/language-models:${VERSION}

bleurt_scorer_bert_scorer:
	./boot.sh $@

gpt2:
	./boot.sh $@

dummy:
	./boot.sh $@

bloom:
	./boot.sh $@

gptneox:
	./boot.sh $@

opt66:
	./boot.sh $@

t0pp:
	./boot.sh $@

alpaca_7b:
	./setup/$@.sh
	./boot.sh $@

pythia:
	./boot.sh $@

oasst:
	# ./setup/$@.sh
	./boot.sh $@

vicuna_7b:
	./setup/$@.sh
	./boot.sh $@

vicuna_13b:
	./setup/$@.sh
	./boot.sh $@

llama_65b:
	./boot.sh $@

llama_30b:
	./boot.sh $@

llama_30b_supercot:
	./boot.sh $@

falcon_40b:
	./boot.sh $@

falcon_40b_instruct:
	./boot.sh $@

baize_7b:
	./boot.sh $@

baize_13b:
	./boot.sh $@
