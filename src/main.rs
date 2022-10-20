use ndarray::prelude::*;
use ndarray_rand::{RandomExt, SamplingStrategy};
use ndarray_rand::rand as rand;
use rand::seq::IteratorRandom;

// Stats
use ndarray_rand::rand_distr::{Uniform, StandardNormal}; 

use ndarray_stats::HistogramExt;
use ndarray_stats::histogram::{strategies::Sqrt, GridBuilder};
use noisy_float::types::{N64, n64};


use hypermelon::format_move;

fn main(){
    let arr1 = array![1.,2.,3.,4.,5.,6.];
    println!("1D array : {}",arr1);

    let arr2 = array![1., 2.2, 3.3, 4., 5., 6.];
    let arr3 = arr1 + arr2;

    println!("Sum : {}", arr3);

    let arr4 = array![
        [1., 2., 3.], 
        [ 4., 5., 6.]
        ];
    let arr5 = Array::from_elem((2, 1), 1.);
    let arr6 = arr4 + arr5;
    println!("2D array:\n{}", arr6);

    let arr7 =  Array::<f64, _>::zeros(arr6.raw_dim());
    let arr8 = arr6 * arr7;
    println!("\n{}", arr8);

    let identity: &Array2<f64> = &Array::eye(3);
    println!("\n{}", identity);

    let arr9 = array![[1., 2., 3.], [ 4., 5., 6.], [7., 8., 9.]];
    let arr10 = &arr9 * identity;
    println!("\n{}", arr10);

    let arr11 = arr9.dot(identity);
    println!("\n{}", arr11);

    let arr12 = Array::<i8, _>::ones((2, 3, 2, 2));
    println!("\nMULTIDIMENSIONAL\n{}", arr12);

    let arr13 = Array::random((2, 5), Uniform::new(0., 10.));
    println!("{:5.2}", arr13);

    let arr14 = array![1., 2., 3., 4., 5., 6.];
    let arr15 = arr14.sample_axis(Axis(0), 2, SamplingStrategy::WithoutReplacement);
    println!("\nSampling from:\t{}\nTwo elements:\t{}", arr14, arr15);

    let mut rng = rand::thread_rng();
    let faces = "😀😎😐😕😠😢";
    let arr16 = Array::from_shape_vec((2, 2), faces.chars().choose_multiple(&mut rng, 4)).unwrap();
    println!("\nSampling from:\t{}", faces);
    println!("Elements:\n{}", arr16);

    // Stats

    // First try
    let arr17 = Array::<f64, _>::random_using((10000,2), StandardNormal, &mut rand::thread_rng());

    let data = arr17.mapv(|e| n64(e));

    let grid = GridBuilder::<Sqrt<N64>>::from_array(&data).unwrap().build();
    let histogram = data.histogram(grid);
    let histogram_matrix = histogram.counts();
    let data = histogram_matrix.sum_axis(Axis(0));
    let his_data: Vec<(f64, f64)> = data.iter().enumerate().map(|(e, i)| (e as f64, *i as f64) ).collect();

    let file = std::fs::File::create("standard_normal_hist.svg").unwrap();

    // let graph = poloto::build::plot("Histogram");
    // graph.histogram().cloned(his_data.iter());

    println!("{}",his_data.len());

    // let size = f64::try_from(his_data.len()).unwrap();

    // let plots = poloto::plots!(
    //     poloto::build::plot("Histogram").histogram().cloned(his_data.iter()),
    //     poloto::build::markers([size], [])
    // );

    // Second try
    let file = std::fs::File::create("standard_normal_hist_2.svg").unwrap();

    let trend = vec![
        0, 0, 0, 0, 0, 3, 5, 5, 10, 20, 50, 60, 70, 50, 40, 34, 34, 20, 10, 20, 10, 4, 2, 0,
    ];

    let size = i128::try_from(trend.len()).unwrap();

    let it = (0..).zip(trend.iter().copied());

    let plots = poloto::plots!(
        poloto::build::plot("").histogram().cloned(it),
        poloto::build::markers([size], [])
    );

    let data = poloto::data(plots);

    let ticks =
        poloto::ticks::from_iter((0..).step_by(6)).with_tick_fmt(|&v| format_move!("{} hr", v));

    data.map_xticks(|_| ticks)
        .build_and_label(("title", "x", "y"))
        .append_to(poloto::header().light_theme())
        .render_io_write(file).unwrap();


}