use std::array;

fn education(education: &str) -> f64 {
    match education {
        "university" => 3.0,
        "graduate school" => 2.0,
        "high school" => 1.0,
        _ => 1.0,
    }
}

fn status(status: &str) -> f64 {
    match status {
        "married" => 1.0,
        "single" => 0.0,
        _ => 0.0,
    }
}

fn parse_row(row: &str) -> ([f64; 22], Option<u8>) {
    let mut result = [0.0; 22];
    let mut iter = row.split(',');
    iter.next(); //skip id
    result[0] = iter.next().unwrap().parse::<f64>().unwrap();
    iter.next(); //skip sex
    result[1] = education(iter.next().unwrap());
    result[2] = status(iter.next().unwrap());
    let other_data: [f64; 19] = array::from_fn(|_| iter.next().unwrap().parse::<f64>().unwrap());
    result[3..].copy_from_slice(&other_data);
    let y = iter.next().and_then(|x| x.parse::<u8>().ok());
    (result, y)
}

pub fn parse_rows() -> (Vec<Vec<f64>>, Vec<u8>) {
    let file = include_str!("../data_participant.csv");
    let mut lines = file.lines();
    lines.next();
    let mut x = vec![];
    let mut y = vec![];
    for row in lines {
        let result = parse_row(row);
        if let Some(data) = result.1 {
            x.push(result.0.into_iter().collect());
            y.push(data);
        }
    }
    (x, y)
}

pub fn result() -> Vec<Vec<f64>> {
    let file = include_str!("../desjardins_presubmissions.csv");
    let mut lines = file.lines();
    lines.next();
    let mut x = vec![];
    for row in lines {
        let mut row: String = row.chars().skip_while(|&c| c != ',').skip(1).collect();
        let result = parse_row(&row);
        if let None = result.1 {
            x.push(result.0.into_iter().collect());
        }
    }
    x
}
