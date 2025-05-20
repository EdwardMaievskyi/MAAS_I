FROM python:3.10-slim

WORKDIR /app
RUN pip install pandas numpy scikit-learn
# This environment will be used to install specific libraries requested by tasks
# and then run the Python code.
CMD ["python"]
