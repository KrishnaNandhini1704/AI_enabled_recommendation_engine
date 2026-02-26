# Definition of Done (DoD)

All tasks and user stories must meet these criteria before being moved to "Done".

### 1. Code Quality
- [ ] Code follows PEP 8 standards (for Python).
- [ ] Variables and functions have descriptive names.
- [ ] No hardcoded paths relative to the developer's machine.

### 2. Testing
- [ ] Unit tests pass for the changed logic.
- [ ] No regressions in the recommendation output.
- [ ] UI is tested on at least Chrome and Mobile view.

### 3. Documentation
- [ ] `README.md` is updated if there are new setup steps.
- [ ] Backlog item is updated with the resolution summary.
- [ ] Any new env variables are documented.

### 4. Machine Learning
- [ ] Model metrics (RMSE, Variance) are documented.
- [ ] Inference time is under 500ms.
