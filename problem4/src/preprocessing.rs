use smartcore::{
    ensemble::random_forest_classifier::{
        RandomForestClassifier, RandomForestClassifierParameters,
    },
    linalg::basic::matrix::DenseMatrix,
    naive_bayes::{
        bernoulli::{BernoulliNB, BernoulliNBParameters},
        gaussian::{GaussianNB, GaussianNBParameters},
    },
    tree::decision_tree_classifier::SplitCriterion,
};
use std::array;

type DataX = [f64; 20];
type DataY = u8;

#[derive(Debug, PartialEq)]
pub enum Education {
    University,
    GraduateSchool,
    HighSchool,
    Other,
    Unknown,
}

impl Education {
    fn new(education: &str) -> Self {
        match education {
            "university" => Self::University,
            "graduate school" => Self::GraduateSchool,
            "high school" => Self::HighSchool,
            _ => Self::Unknown,
        }
    }
    fn to_index(&self) -> usize {
        match self {
            Self::University => 0,
            Self::GraduateSchool => 1,
            Self::HighSchool => 2,
            Self::Other => 3,
            Self::Unknown => 4,
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum MaritalStatus {
    Married,
    Single,
    Other,
    Unknown,
}

impl MaritalStatus {
    fn new(status: &str) -> Self {
        match status {
            "single" => Self::Single,
            "married" => Self::Married,
            "other" => Self::Other,
            _ => Self::Unknown,
        }
    }
    fn to_index(&self) -> usize {
        match self {
            Self::Single => 0,
            Self::Married => 1,
            Self::Other => 2,
            Self::Unknown => 3,
        }
    }
}

#[derive(Debug)]
pub struct Models {
    models: [[Model; 4]; 5],
}

impl Models {
    pub fn from_csv() -> Models {
        let file = include_str!("../data_participant.csv");
        let mut lines = file.lines();
        lines.next();
        let mut models = Models::new();
        for row in lines {
            let (education, marital_status, x, y) = parse_row(row);
            if let Some(y) = y {
                models.push(education, marital_status, x, y);
            }
        }
        models
    }
    fn new() -> Self {
        Self {
            models: array::from_fn(|_| array::from_fn(|_| Model::new())),
        }
    }
    fn push(&mut self, education: Education, marital_status: MaritalStatus, x: DataX, y: DataY) {
        if education != Education::Unknown {
            if marital_status != MaritalStatus::Unknown {
                self.models[4][3].push(x, y);
            }
            self.models[4][marital_status.to_index()].push(x, y);
        }
        self.models[education.to_index()][marital_status.to_index()].push(x, y);
    }
    pub fn train(self) -> TrainedModels {
        let models = self.models.map(|layer| layer.map(Model::train));
        TrainedModels { models }
    }
}

#[derive(Debug)]
struct Model {
    data: (Vec<DataX>, Vec<DataY>),
}

impl Model {
    fn new() -> Self {
        Self {
            data: (vec![], vec![]),
        }
    }
    fn push(&mut self, x: DataX, y: DataY) {
        self.data.0.push(x);
        self.data.1.push(y);
    }
    fn split_data(&self, ratio: f64) -> ((Vec<Vec<f64>>, Vec<u8>), (Vec<Vec<f64>>, Vec<u8>)) {
        let training_row = (self.data.0.len() as f64 * ratio) as usize;
        let (x_train, x_ref) = self.data.0.split_at(training_row);
        let (x_train, x_ref) = (
            x_train.iter().map(|a| a.to_vec()).collect(),
            x_ref.iter().map(|a| a.to_vec()).collect(),
        );
        let (y_train, y_ref) = self.data.1.split_at(training_row);
        let (y_train, y_ref) = (
            y_train.iter().copied().collect(),
            y_ref.iter().copied().collect(),
        );
        ((x_train, y_train), (x_ref, y_ref))
    }
    fn train(self) -> TrainedModel {
        if self.data.0.len() < 20 {
            return TrainedModel::Uncommon;
        }
        let ((x_train, y_train), (x_test, y_test)) = self.split_data(0.92);
        let len = x_train.len();
        let x_train = DenseMatrix::from_2d_vec(&x_train);
        let x_test = DenseMatrix::from_2d_vec(&x_test);
        let (y_test_result, trained_model) = if len > 20000 {
            println!("Training the model gaussian");
            let trained_model =
                GaussianNB::fit(&x_train, &y_train, GaussianNBParameters::default()).unwrap();
            let y_test_result = trained_model.predict(&x_test).unwrap();
            (y_test_result, TrainedModel::GaussianNB(trained_model))
        } else if len > 2000 {
            println!("Training the model binomial");
            let trained_model =
                BernoulliNB::fit(&x_train, &y_train, BernoulliNBParameters::default()).unwrap();
            let y_test_result = trained_model.predict(&x_test).unwrap();
            (y_test_result, TrainedModel::BernoulliNB(trained_model))
        } else {
            println!("Training the model random forest");
            let trained_model = RandomForestClassifier::fit(
                &x_train,
                &y_train,
                RandomForestClassifierParameters::default()
                    .with_n_trees(100)
                    .with_min_samples_leaf(1)
                    .with_criterion(SplitCriterion::Entropy)
                    .with_max_depth(100)
                    .with_m(50),
            )
            .unwrap();
            let y_test_result = trained_model.predict(&x_test).unwrap();
            (y_test_result, TrainedModel::RandomForest(trained_model))
        };

        println!("Finishing the training");

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
        trained_model
    }
}

enum TrainedModel {
    RandomForest(RandomForestClassifier<f64, u8, DenseMatrix<f64>, Vec<u8>>),
    BernoulliNB(BernoulliNB<f64, u8, DenseMatrix<f64>, Vec<u8>>),
    GaussianNB(GaussianNB<f64, u8, DenseMatrix<f64>, Vec<u8>>),
    Uncommon,
}

impl TrainedModel {
    fn predict(&self, x: &[f64]) -> u8 {
        let x = DenseMatrix::from_2d_array(&vec![x]);
        match &self {
            Self::RandomForest(model) => model.predict(&x),
            Self::GaussianNB(model) => model.predict(&x),
            Self::BernoulliNB(model) => model.predict(&x),
            Self::Uncommon => panic!("PANIQUE"),
        }
        .unwrap()[0]
    }
}

pub struct TrainedModels {
    models: [[TrainedModel; 4]; 5],
}

impl TrainedModels {
    pub fn predict(&self, rows: &[(Education, MaritalStatus, Vec<f64>)]) -> Vec<u8> {
        let mut y = vec![];
        for (education, marital_status, x) in rows {
            let new_y = self.models[education.to_index()][marital_status.to_index()].predict(x);
            y.push(new_y)
        }
        y
    }
}

fn parse_row(row: &str) -> (Education, MaritalStatus, DataX, Option<DataY>) {
    let mut result = DataX::default();
    let mut iter = row.split(',');
    iter.next(); //skip id
    result[0] = iter.next().unwrap().parse::<f64>().unwrap();
    iter.next().unwrap(); //skip sex
    let education = Education::new(iter.next().unwrap());
    let marital_status = MaritalStatus::new(iter.next().unwrap());

    let other_data: [f64; 19] = array::from_fn(|_| (iter.next().unwrap().parse::<f64>().unwrap()));
    result[1..].copy_from_slice(&other_data);
    let y = iter.next().and_then(|x| x.parse::<u8>().ok());
    (education, marital_status, result, y)
}

pub fn result() -> Vec<(Education, MaritalStatus, Vec<f64>)> {
    let file = include_str!("../desjardins_presubmissions.csv");
    let mut lines = file.lines();
    lines.next();
    let mut to_process = vec![];
    for row in lines {
        let row: String = row.chars().skip_while(|&c| c != ',').skip(1).collect();
        let (education, marital_status, x, _) = parse_row(&row);
        to_process.push((education, marital_status, x.into_iter().collect()));
    }
    to_process
}
