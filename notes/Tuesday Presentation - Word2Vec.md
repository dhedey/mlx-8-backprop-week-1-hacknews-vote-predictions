
## Next steps

* Investigate tokenizers, word2vec
* [Weights & Biases - Sweeps](https://docs.wandb.ai/guides/sweeps/) docs to optimize hyper parameters
* Discuss how to handle unknown users or unknown domains
	* Perhaps using a user / domain embedding with a bias, with a Laplace weight distribution to encourage the weights to be close to 0; so that most of the users that don't come up much in the samples will get an embedding close to the bias.