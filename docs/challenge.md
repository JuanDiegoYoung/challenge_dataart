# Challenge Notes

## Setup
- Created the repository and kept the original challenge structure.
- Added a minimal Python .gitignore.
- Worked with main, develop, and feature/part1-model branches.

## Initial issues
- Python 3.14 caused dependency installation problems, so I standardized the environment on Python 3.10.
- The test stack had a dependency compatibility issue, which I fixed by pinning anyio to <4.
- After fixing the environment, the first code-level failure appeared in challenge/model.py due to an invalid typing annotation.
