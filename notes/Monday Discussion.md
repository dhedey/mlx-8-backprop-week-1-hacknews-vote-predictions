# Questions / Explorations

* Distribution of upvotes?
* Re-working data into useful data
* What is word2vec?
* How do we extract TLD+1 (i.e. domain or `github.com/yuuki`)
* How do we convert word embeddings into an embedding for the title?
	* Do we want cluster analysis on the topic of the title information? What does this buy us?
	* Would it be helpful to learn topics? Maybe word2vec can give us this?
	* It depends on how the title embedding can work?

# Ideas

* "Only learn from the best" - compare the title of a new submission to the "top X titles" to give a similarity score.
	* Might be useful; might miss some things

# Data filtering

```sql
type = 'story'  -- Ignore comments; jobs; etc
```
# Embeddings

### User Data

* Karma (upvotes - downvotes)
* When were they created?
* Distribution of upvotes per post
	* Rather than just relying on karma
	* (e.g. Upvotes / Post; or some other wider sense of the distribution)


### Items

Item = `story` is the main type of post on the site.

Useful data
* Title embedding
	* How long is the title? (possibly learnt? or explicit feature?)
	* Word meanings (need to investigate CBOW + word2vec)
* File type (URL ending)
	* Website? PDF?
* URL Domain (Bes recommended extracting the domain from the URL)
	* Feature/s based on the distribution of upvotes of that domain's posts
* Author
	* Feature/s based on the distribution of upvotes of that author's posts
* Date
	* We could consider only training on newer posts; because what's popular might change
		* OR we could scale/bias the training towards more recent data (?)
	* Or we try to model the date into the data; but this might not be so helpful