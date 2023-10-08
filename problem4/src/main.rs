pub mod preprocessing;

use ndarray_rand::rand::SeedableRng;
use rand::rngs::SmallRng;

use crate::preprocessing::{parse_rows, result};
use linfa::prelude::*;
use linfa_svm::{error::Result, Svm};

fn main() -> Result<()> {
    let mut rng = SmallRng::seed_from_u64(42);

    let (train, test) = parse_rows().shuffle(&mut rng).split_with_ratio(0.9);

    println!("Training model with Gini criterion ...");
    let model = Svm::<_, bool>::params()
        .pos_neg_weights(10., 10.)
        .gaussian_kernel(80.0)
        .fit(&train)?;

    let gini_pred_y = model.predict(&test);
    let cm = gini_pred_y.confusion_matrix(&test)?;

    let to_predict = model.predict(&result());
    let mut string = String::new();
    let mut lines = include_str!("../desjardins_presubmissions.csv").lines();
    lines.next();
    string.push_str(&("ID,default-payment-next-month\n"));
    for (row, value) in lines.zip(to_predict) {
        let mut tests = row.split(',');
        tests.next();
        let id = tests.next().unwrap();
        if value {
            string.push_str(&format!("{},1\n", id));
        } else {
            string.push_str(&format!("{},0\n", id));
        }
    }
    std::fs::write("test.csv", string).unwrap();

    println!("{:?}", cm);

    println!(
        "Test accuracy with Gini criterion: {:.2}%",
        100.0 * cm.accuracy()
    );

    // let gini_pred_y = gini_model.predict();

    Ok(())
}
