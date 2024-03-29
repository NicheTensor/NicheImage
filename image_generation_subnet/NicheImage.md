# NicheImage - High Quality Images Tailored to Your Need


## Uniqueness of the subnet, contribution to the overall Bittensor environment and ML community, long-term vision (40%)

This subnet offers large scale inference of machine learning models to be used in production.

This can greatly reduce costs of generating data and using live APIs by utilizing decentralized compute. The vision is to make it possible for unused hardware to easily join the network and produce value, instead of being underutilized.

We make it possible to support multiple use cases by having a modular structure that allows miners in different categories be scored in different ways, making it possible to easily add and remove categories based on demand.

A category can be for running a specific model doing a specific set of task. For example the DreamShaper category tells miners to run the DreamShaper model and support the tasks text-2-image, image-2-image and ControlNet which takes both a text and an image as input and transforms the image based on the input text.

The miners choosing that category is then tested by comparing the generated result with the expacted result a fraction of the time, by doing a similarity measure of the embeddings of the image generated by the miner compared with the embeddings of the image generated by the validator.

A miner only needs to be evaluated a fraction of the time, meaning that the validator can save compute by rarely generating images themselves.

You can test out current models available at: https://nicheimage.streamlit.app/



## Subnet design, robustness of the reward mechanism, design of the miners and validators (40%)

The reward mechanism is simple. Validators first query miners asking them what model they are running. Validators then proceed to send text prompts and seeds to miners, to which they need to generate an image. The image is returned and the validator can check if the miner is running the image model they claimed by also running the same model and comparing the image embeddings.

If the image embeddings are nearly identical, the miner scores 1 - speed penalty, and the speed penalty is calculated  0.4  * processing_time^3 / 12^3

This incentivizes consistently fast response times, since the penalty grows polinomially with time.

To make sure that prompt data never runs out, a combination of random seeds and a prompt generation model is used by the validator. Further as organic queries to the network grows, it naturally adds diversity to the requests sent to miners.

Further, the miners rate limit validators based on Tao, ensuring that no validator get to use the network disproportionally much, and dissincentivizes miners from blacklisting even small miners.


## Community engagement, overall response to feedback, and subnet stability (20%)

We work closely with the community to make development as smooth and predictable as possible.

Therefore we have a set time each week, Tuesday at 19:00 CET time, where new updates are pushed. However not all weeks have an update since we don't want to push too small updates. 

This predictability allows miners and validators to plan ahead accordingly, and we can all communicate and quickly find out if there is any issue with an update, and roll back if neccessary (which we chose to do on one occasion rather than trying to push a hotfix that may or may not have solved the problem for everyone).