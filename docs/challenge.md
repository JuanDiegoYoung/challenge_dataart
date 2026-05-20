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

⚠️ Mistakes:
- Forgot to do a PR for the merging of PART 1. I used a no-fast-forward merge into develop (and kept the feature branch). For the following parts, I will use PRs to make the integration flow explicit.

## Part 2

- Implemented the FastAPI /predict endpoint in challenge/api.py.
- The API loads and trains the selected model once when the app starts.
- The endpoint validates OPERA, TIPOVUELO and MES before predicting.
- Invalid flight payloads return HTTP 400.
- make api-test is passing.

## Part 3

- Deployed the API to Google Cloud Run using the Dockerfile in the repository.
- Updated the Makefile stress URL to the public Cloud Run service URL.
- Deployed URL: https://challenge-dataart-api-68330138622.europe-west1.run.app
- Health check on the deployed service is responding correctly.
- make stress-test is passing against the deployed service.

Issues found:
- The initial Cloud Run deployment failed because the default build service account was missing the specific Cloud Run build role.
- The local stress-test environment also needed additional version pins for the old locust stack: Jinja2, Werkzeug and itsdangerous.

## Part 4

- Added GitHub Actions workflows under .github/workflows.
- CI runs on push and pull request, installs dependencies and executes make model-test and make api-test.
- CD deploys automatically to Google Cloud Run on pushes to develop.
- The deployment workflow requires a repository secret named GCP_SA_KEY with a service account key that has permission to deploy to Cloud Run.

