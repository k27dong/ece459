#![warn(clippy::all)]
use crossbeam::channel::{unbounded, Receiver, Sender};
use lab4::idea::Idea;
use lab4::package::Package;
use lab4::{
    checksum::Checksum, idea::IdeaGenerator, package::PackageDownloader, student::Student,
    PrintInfo,
};
use std::env;
use std::fs;
use std::error::Error;
use std::sync::{Arc, Mutex};
use std::thread::spawn;

struct Args {
    pub num_ideas: usize,
    pub num_idea_gen: usize,
    pub num_pkgs: usize,
    pub num_pkg_gen: usize,
    pub num_students: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().collect();
    let num_ideas = args.get(1).map_or(Ok(80), |a| a.parse())?;
    let num_idea_gen = args.get(2).map_or(Ok(2), |a| a.parse())?;
    let num_pkgs = args.get(3).map_or(Ok(4000), |a| a.parse())?;
    let num_pkg_gen = args.get(4).map_or(Ok(6), |a| a.parse())?;
    let num_students = args.get(5).map_or(Ok(6), |a| a.parse())?;
    let args = Args {
        num_ideas,
        num_idea_gen,
        num_pkgs,
        num_pkg_gen,
        num_students,
    };

    hackathon(&args);

    Ok(())
}

fn per_thread_amount(thread_idx: usize, total: usize, threads: usize) -> usize {
    let per_thread = total / threads;
    let extras = total % threads;
    per_thread + (thread_idx < extras) as usize
}

fn print_jobs(rx: Receiver<PrintInfo>, num_student: usize) {
    let mut counter = 0;
    loop {
        let info = rx.recv().unwrap();

        if (info.id == 0) && (info.name == "DONE") {
            counter += 1;
            if counter == num_student {
                break;
            }
        } else {
            println!(
                "\nStudent {} built {} using {} packages\nIdea checksum: {}\nPackage checksum: {}",
                info.id, info.name, info.num_pkg_required, info.idea_checksum, info.pkg_checksum
            );

            for pkg in info.pkg_used.iter() {
                println!("> {}", pkg.name);
            }
        }
    }
}

fn hackathon(args: &Args) {
    // file read
    let f_path_pkg: &str = "data/packages.txt";
    let f_path_ideas_products: &str = "data/ideas-products.txt";
    let f_path_ideas_customers: &str = "data/ideas-customers.txt";

    let pkg_raw_arr = fs::read_to_string(f_path_pkg).unwrap();
    let idea_product_raw_arr = fs::read_to_string(f_path_ideas_products).unwrap();
    let idea_customer_raw_arr = fs::read_to_string(f_path_ideas_customers).unwrap();

    let pkg_arr = Arc::new(
        pkg_raw_arr
            .lines()
            .map(|s| s.to_string())
            .collect::<Vec<String>>(),
    );

    let idea_cross_product: Vec<(String, String)> = idea_product_raw_arr
        .lines()
        .flat_map(|p| {
            idea_customer_raw_arr
                .lines()
                .map(move |c| (p.to_owned(), c.to_owned()))
        })
        .collect();

    let idea_arr = Arc::new(idea_cross_product);

    // Create channels, threads, and shared checksums
    let (print_send, print_recv) = unbounded::<PrintInfo>();
    let (pkg_send, pkg_recv) = unbounded::<Package>();
    let (idea_send, idea_recv) = unbounded::<Idea>();
    let (out_of_idea_send, out_of_idea_recv) = unbounded::<bool>();
    let mut threads = vec![];
    let mut idea_checksum = Arc::new(Mutex::new(Checksum::default()));
    let mut pkg_checksum = Arc::new(Mutex::new(Checksum::default()));
    let mut student_idea_checksum = Arc::new(Mutex::new(Checksum::default()));
    let mut student_pkg_checksum = Arc::new(Mutex::new(Checksum::default()));
    let num_student = args.num_students;

    // Spawn print job thread
    threads.push(spawn(move || print_jobs(print_recv, num_student)));

    // Spawn student threads
    for i in 0..args.num_students {
        let mut student = Student::new(
            i,
            Receiver::clone(&idea_recv),
            Receiver::clone(&pkg_recv),
            Receiver::clone(&out_of_idea_recv),
            Sender::clone(&print_send),
        );
        let student_idea_checksum = Arc::clone(&student_idea_checksum);
        let student_pkg_checksum = Arc::clone(&student_pkg_checksum);
        let thread = spawn(move || student.run(student_idea_checksum, student_pkg_checksum));
        threads.push(thread);
    }

    // Spawn package downloader threads. Packages are distributed evenly across threads.
    let mut start_idx = 0;
    for i in 0..args.num_pkg_gen {
        let num_pkgs = per_thread_amount(i, args.num_pkgs, args.num_pkg_gen);
        let downloader = PackageDownloader::new(start_idx, num_pkgs, Sender::clone(&pkg_send));
        let pkg_checksum = Arc::clone(&pkg_checksum);
        start_idx += num_pkgs;

        let shared_pkg_list = Arc::clone(&pkg_arr);

        let thread = spawn(move || downloader.run(pkg_checksum, shared_pkg_list));
        threads.push(thread);
    }
    assert_eq!(start_idx, args.num_pkgs);

    // Spawn idea generator threads. Ideas and packages are distributed evenly across threads. In
    // each thread, packages are distributed evenly across ideas.
    let mut start_idx = 0;
    for i in 0..args.num_idea_gen {
        let shared_idea_list = Arc::clone(&idea_arr);
        let num_ideas = per_thread_amount(i, args.num_ideas, args.num_idea_gen);
        let num_pkgs = per_thread_amount(i, args.num_pkgs, args.num_idea_gen);
        let num_students = per_thread_amount(i, args.num_students, args.num_idea_gen);
        let generator = IdeaGenerator::new(
            start_idx,
            num_ideas,
            num_students,
            num_pkgs,
            Sender::clone(&idea_send),
            Sender::clone(&out_of_idea_send),
        );
        let idea_checksum = Arc::clone(&idea_checksum);
        start_idx += num_ideas;

        let thread = spawn(move || generator.run(idea_checksum, shared_idea_list));
        threads.push(thread);
    }
    assert_eq!(start_idx, args.num_ideas);

    // Join all threads
    threads.into_iter().for_each(|t| t.join().unwrap());

    let idea = Arc::get_mut(&mut idea_checksum).unwrap().get_mut().unwrap();
    let student_idea = Arc::get_mut(&mut student_idea_checksum)
        .unwrap()
        .get_mut()
        .unwrap();

    let pkg = Arc::get_mut(&mut pkg_checksum).unwrap().get_mut().unwrap();
    let student_pkg = Arc::get_mut(&mut student_pkg_checksum)
        .unwrap()
        .get_mut()
        .unwrap();

    println!("Global checksums:\nIdea Generator: {}\nStudent Idea: {}\nPackage Downloader: {}\nStudent Package: {}",
        idea, student_idea, pkg, student_pkg);
}
