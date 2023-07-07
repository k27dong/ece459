use super::{checksum::Checksum, idea::Idea, package::Package, PrintInfo};
use crossbeam::channel::{Receiver, Sender};
use std::sync::{Arc, Mutex};

pub struct Student {
    id: usize,
    idea: Option<Idea>,
    pkgs: Vec<Package>,
    idea_receiver: Receiver<Idea>,
    pkg_receiver: Receiver<Package>,
    out_of_ideas_receiver: Receiver<bool>,
    print_sender: Sender<PrintInfo>,
}

impl Student {
    pub fn new(
        id: usize,
        idea_receiver: Receiver<Idea>,
        pkg_receiver: Receiver<Package>,
        out_of_ideas_receiver: Receiver<bool>,
        print_sender: Sender<PrintInfo>,
    ) -> Self {
        Self {
            id,
            idea: None,
            pkgs: vec![],
            idea_receiver,
            pkg_receiver,
            out_of_ideas_receiver,
            print_sender,
        }
    }

    fn build_idea(
        &mut self,
        idea_checksum: &Arc<Mutex<Checksum>>,
        pkg_checksum: &Arc<Mutex<Checksum>>,
    ) {
        if let Some(ref idea) = self.idea {
            let pkgs_required = idea.num_pkg_required;
            let pkgs_used = self.pkgs.drain(0..pkgs_required).collect::<Vec<_>>();

            // Update idea and package checksums
            // All of the packages used in the update are deleted, along with the idea
            let mut temp_pkg_checksum = Checksum::default();
            for pkg in pkgs_used.iter() {
                temp_pkg_checksum.update(Checksum::with_sha256(&pkg.name));
            }

            let mut idea_checksum = idea_checksum.lock().unwrap();
            idea_checksum.update(Checksum::with_sha256(&idea.name));
            let mut pkg_checksum = pkg_checksum.lock().unwrap();
            pkg_checksum.update(temp_pkg_checksum);

            // Send printing information to the printer
            self.print_sender
                .send(PrintInfo {
                    id: self.id,
                    name: idea.name.clone(),
                    num_pkg_required: pkgs_required,
                    idea_checksum: idea_checksum.to_string(),
                    pkg_checksum: pkg_checksum.to_string(),
                    pkg_used: pkgs_used,
                })
                .unwrap();

            self.idea = None;
        }
    }

    pub fn run(&mut self, idea_checksum: Arc<Mutex<Checksum>>, pkg_checksum: Arc<Mutex<Checksum>>) {
        loop {
            let new_idea = self.idea_receiver.try_recv();

            if new_idea.is_err()
                && self.idea_receiver.is_empty()
                && self.out_of_ideas_receiver.try_recv().is_ok()
            {
                self.print_sender
                    .send(PrintInfo {
                        id: 0,
                        name: String::from("DONE"),
                        num_pkg_required: 0,
                        idea_checksum: String::new(),
                        pkg_checksum: String::new(),
                        pkg_used: vec![],
                    })
                    .unwrap();
                return;
            }

            // If the student receives a new idea, they will attempt to build it
            if new_idea.is_ok() {
                self.idea = Some(new_idea.unwrap());
                let num_pkg_required = self.idea.as_ref().unwrap().num_pkg_required;

                // since by design there will always be enough packages to build an idea, so once the
                // student receives a new idea, they will keep looking for pkgs in the pkg channel until
                // they suddenly have enough pkgs to build the idea
                // (in most cases, the speed would be bottlenecked by the idea channel, but let's be safe
                // and still add the try_recv)
                loop {
                    if self.pkgs.len() >= num_pkg_required {
                        self.build_idea(&idea_checksum, &pkg_checksum);
                        break;
                    }

                    let new_pkg = self.pkg_receiver.try_recv();
                    if new_pkg.is_ok() {
                        self.pkgs.push(new_pkg.unwrap());
                    }
                }
            }
        }
    }
}
