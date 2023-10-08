use linfa::Dataset;
use ndarray::{Array, Array1, Array2, Ix1};
use std::{array, collections::HashMap};

enum Education {
    University,
    GraduateSchool,
    HighSchool,
}

impl Education {
    fn from(education: &str) -> Self {
        match education {
            "university" => Self::University,
            "graduate school" => Self::GraduateSchool,
            "high school" => Self::HighSchool,
            _ => Self::HighSchool,
        }
    }
}

enum Status {
    Married,
    Single,
}

impl Status {
    fn from(status: &str) -> Self {
        match status {
            "married" => Self::Married,
            "single" => Self::Single,
            _ => Self::Single,
        }
    }
}

fn parse_row(row: &str) -> (Education, Status, [f64; 19], Option<bool>) {
    let mut result = [0.0; 19];
    let mut iter = row.split(',');
    iter.next(); //skip id
    result[0] = iter.next().unwrap().parse::<f64>().unwrap();
    iter.next(); //skip sex
    let education = Education::from(iter.next().unwrap());
    let status = Status::from(iter.next().unwrap());
    let other_data: [f64; 18] = array::from_fn(|_| iter.next().unwrap().parse::<f64>().unwrap());
    result[3..].copy_from_slice(&other_data);
    let y = iter
        .next()
        .and_then(|x| x.parse::<u8>().ok())
        .map(|x| x != 0);
    (education, status, result, y)
}

pub fn parse_rows() -> Dataset<f64, bool, Ix1> {
    let file = include_str!("../data_participant.csv");
    let mut lines = file.lines();
    lines.next();
    let mut x = vec![];
    let mut y = vec![];
    let mut data = HashMap::new();
    for row in lines {
        let result = parse_row(row);
        if let Some(data) = result.3 {
            data.
            // x.extend(&result.0);
            y.push(data);
        }
    }
    let x: Array2<f64> = Array::from_shape_vec((x.len() / 21, 21), x).unwrap();
    let y: Array1<bool> = Array::from_vec(y);
    Dataset::new(x, y)
}

pub fn result() -> Array2<f64> {
    let file = include_str!("../desjardins_presubmissions.csv");
    let mut lines = file.lines();
    lines.next();
    let mut x = vec![];
    for row in lines {
        let result = parse_row(row);
        x.extend(&result.0);
    }
    Array::from_shape_vec((x.len() / 21, 21), x).unwrap()
}
