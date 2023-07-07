use super::checksum::Checksum;
use crossbeam::channel::Sender;
use std::sync::{Arc, Mutex};

pub struct Package {
    pub name: String,
}

pub struct PackageDownloader {
    pkg_start_idx: usize,
    num_pkgs: usize,
    pkg_sender: Sender<Package>,
}

impl PackageDownloader {
    pub fn new(pkg_start_idx: usize, num_pkgs: usize, pkg_sender: Sender<Package>) -> Self {
        Self {
            pkg_start_idx,
            num_pkgs,
            pkg_sender,
        }
    }

    pub fn run(&self, pkg_checksum: Arc<Mutex<Checksum>>, pkg_arr: Arc<Vec<String>>) {
        // Generate a set of packages and place them into the event queue
        // Update the package checksum with each package name
        let mut temp_checksum = Checksum::default();

        for i in 0..self.num_pkgs {
            let name = pkg_arr[(self.pkg_start_idx + i) % pkg_arr.len()].clone();
            temp_checksum.update(Checksum::with_sha256(&name));

            self.pkg_sender.send(Package { name }).unwrap();
        }

        pkg_checksum.lock().unwrap().update(temp_checksum);
    }
}
