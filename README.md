# BNN symmetry breaking

See this [article](https://tavin.io/article.html) for a brief explanation.

### Running an experiment

    python run_experiment.py --dataset=BOSTON_HOUSING --model=reference_model --hidden_dim=30 --outdir=output

You get pickled output including the final `mcmc` state which you can dissect as you please.
