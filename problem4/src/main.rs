pub mod preprocessing;

use crate::preprocessing::{result, Models};

fn main() {
    let models = Models::from_csv().train();
    let model_prediction = models.predict(&result());

    let mut string = String::new();
    let mut lines = include_str!("../desjardins_presubmissions.csv").lines();

    lines.next();
    string.push_str(&("ID,default-payment-next-month\n"));
    for (row, value) in lines.zip(model_prediction) {
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
}
