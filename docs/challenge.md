# Challenge Notes

## Setup
- Created the repository and kept the original challenge structure.
- Added a minimal Python .gitignore.
- Worked with main, develop, and feature/part1-model branches.
- Defined virtual environment.
- Run initial tests.
- Model-test started failing on the actual model implementation instead of setup issues.
- End of setup.

## Initial issues
- Python 3.14 caused dependency installation problems, so I standardized the environment on Python 3.10.
- The test stack had a dependency compatibility issue, which I fixed by pinning anyio to <4.
- After fixing the environment, the first code-level failure appeared in challenge/model.py due to an invalid typing annotation.
- Updated the Makefile so the provided tests run from a working directory compatible with the dataset path used in the test suite.
- Removed the broken coverage config reference from the Makefile, since .coveragerc was not present in the repository.

## Part 1

- Implemented preprocessing, training and prediction logic in challenge/model.py.
- To choose the best model, scripts/generate_model_performance.py generates a series of relevant images (confusion matrices, feature importances, etc) which are saved on challenge/model_performance.
- The final chosen model is XGBoost using the top 10 features from the notebook and the class balancing.
- I chose this model because it gave the best recall for delayed flights, which is the most relevant class for this use case.
- Why is the most relevant class? Because the cost of not detecting a delayed flight tends to be higher than the cost of setting one as delayed that, in the end, wasn't. 
- Logistic Regression was simpler and competitive, but the balanced XGBoost model handled the minority class better.
- Model performance plots were saved in challente/model_performance.
- make model-test is passing.

Issues found:
- The notebook uses older seaborn barplot syntax with positional x/y arguments.
- xgboost was used in the notebook but was not listed in the runtime requirements.
