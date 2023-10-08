pub mod preprocessing;

use crate::preprocessing::{parse_rows, result};
use smartcore::{
    ensemble::random_forest_classifier::{
        RandomForestClassifier, RandomForestClassifierParameters,
    },
    linear::logistic_regression::{LogisticRegression, LogisticRegressionParameters},
    naive_bayes::{
        bernoulli::{BernoulliNB, BernoulliNBParameters},
        gaussian::{GaussianNB, GaussianNBParameters},
    },
    neighbors::knn_classifier::{KNNClassifier, KNNClassifierParameters},
    tree::decision_tree_classifier::{DecisionTreeClassifier, DecisionTreeClassifierParameters},
};
use std::error::Error;
// DenseMatrix definition
use smartcore::linalg::basic::matrix::DenseMatrix;
// KNNClassifier

// Various distance metrics

fn split<T: Clone>(test: Vec<T>, ratio: f64) -> (Vec<T>, Vec<T>) {
    let (x1, x2) = test.split_at((ratio * test.len() as f64) as usize);
    (x1.iter().cloned().collect(), x2.iter().cloned().collect())
}

fn main() -> Result<(), Box<dyn Error>> {
    let (x, y) = parse_rows();
    let (x, x_test) = split(x, 0.90);
    let (y, y_test) = split(y, 0.90);
    let x = DenseMatrix::from_2d_vec(&x);
    let x_test = DenseMatrix::from_2d_vec(&x_test);

    let parameters = LogisticRegressionParameters::default();

    let tree = LogisticRegression::fit(&x, &y, parameters).unwrap();

    let next_predict = DenseMatrix::from_2d_vec(&result());
    let to_predict = tree.predict(&next_predict).unwrap();

    let y_test_result = tree.predict(&x_test).unwrap();
    let y_test_result: Vec<u8> = y_test_result.into_iter().map(|c| (c >= 1) as u8).collect();
    println!("{:?}", &y_test[0..42]);
    println!("{:?}", &y_test_result[0..42]);
    let true_positive: u16 = y_test
        .iter()
        .zip(&y_test_result)
        .map(|(&x, &y)| (x == y && x != 0) as u16)
        .sum();

    let false_negative: u16 = y_test
        .iter()
        .zip(&y_test_result)
        .map(|(&x, &y)| (x != y && y == 0) as u16)
        .sum();

    let false_positive: u16 = y_test
        .iter()
        .zip(&y_test_result)
        .map(|(&x, &y)| (x != y && y != 0) as u16)
        .sum();

    let true_negative: u16 = y_test
        .iter()
        .zip(&y_test_result)
        .map(|(&x, &y)| (x == y && x == 0) as u16)
        .sum();
    println!(
        "{} {} {} {}",
        true_positive, true_negative, false_positive, false_negative
    );

    let total = true_negative + false_positive + false_negative + true_positive;
    println!(
        "Accuracy: {}",
        (true_positive + true_negative) as f64 / total as f64
    );

    println!(
        "F1 score {}",
        (2.0 * true_positive as f64)
            / (2.0 * true_positive as f64 + false_positive as f64 + false_negative as f64)
    );

    let mut string = String::new();
    let mut lines = include_str!("../desjardins_presubmissions.csv").lines();

    lines.next();
    string.push_str(&("ID,default-payment-next-month\n"));
    for (row, value) in lines.zip(to_predict) {
        let mut tests = row.split(',');
        tests.next();
        let id = tests.next().unwrap();
        if value == 1 {
            string.push_str(&format!("{},1\n", id));
        } else {
            string.push_str(&format!("{},0\n", id));
        }
    }
    std::fs::write("test.csv", string).unwrap();

    Ok(())
}
