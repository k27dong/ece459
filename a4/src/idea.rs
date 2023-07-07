use super::checksum::Checksum;
use crossbeam::channel::Sender;
use std::sync::{Arc, Mutex};

pub struct Idea {
    pub name: String,
    pub num_pkg_required: usize,
}

pub struct IdeaGenerator {
    idea_start_idx: usize,
    num_ideas: usize,
    num_students: usize,
    num_pkgs: usize,
    idea_sender: Sender<Idea>,
    out_of_ideas_sender: Sender<bool>,
}

impl IdeaGenerator {
    pub fn new(
        idea_start_idx: usize,
        num_ideas: usize,
        num_students: usize,
        num_pkgs: usize,
        idea_sender: Sender<Idea>,
        out_of_ideas_sender: Sender<bool>,
    ) -> Self {
        Self {
            idea_start_idx,
            num_ideas,
            num_students,
            num_pkgs,
            idea_sender,
            out_of_ideas_sender,
        }
    }

    pub fn run(&self, idea_checksum: Arc<Mutex<Checksum>>, idea_arr: Arc<Vec<(String, String)>>) {
        let pkg_per_idea = self.num_pkgs / self.num_ideas;
        let extra_pkgs = self.num_pkgs % self.num_ideas;

        let mut temp_checksum = Checksum::default();

        // Generate a set of new ideas and place them into the event-queue
        // Update the idea checksum with all generated idea names
        for i in 0..self.num_ideas {
            let idea_pair = &idea_arr[(self.idea_start_idx + i) % idea_arr.len()];
            let name = format!("{} for {}", idea_pair.0, idea_pair.1);
            let extra = (i < extra_pkgs) as usize;
            let num_pkg_required = pkg_per_idea + extra;
            let idea = Idea {
                name,
                num_pkg_required,
            };
            temp_checksum.update(Checksum::with_sha256(&idea.name));

            self.idea_sender.send(idea).unwrap();
        }

        idea_checksum.lock().unwrap().update(temp_checksum);

        // Push student termination events into the event queue
        for _ in 0..self.num_students {
            self.out_of_ideas_sender.send(true).unwrap();
        }
    }
}
