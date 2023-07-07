use curl::easy::{Easy2, Handler, WriteError};
use curl::multi::{Easy2Handle, Multi};
use std::str;
use std::sync::atomic::AtomicI32;
use std::time::Duration;

use crate::Packages;

struct Collector(Box<String>);
impl Handler for Collector {
    fn write(&mut self, data: &[u8]) -> Result<usize, WriteError> {
        (*self.0).push_str(str::from_utf8(&data.to_vec()).unwrap());
        Ok(data.len())
    }
}

const DEFAULT_SERVER: &str = "ece459.patricklam.ca:4590";
impl Drop for Packages {
    fn drop(&mut self) {
        self.execute()
    }
}

static _EASYKEY_COUNTER: AtomicI32 = AtomicI32::new(0);

pub struct AsyncState {
    server: String,
    reqs: Vec<Vec<String>>,
}

impl AsyncState {
    pub fn new() -> AsyncState {
        AsyncState {
            server: String::from(DEFAULT_SERVER),
            reqs: Vec::new(),
        }
    }
}

impl Packages {
    pub fn set_server(&mut self, new_server: &str) {
        self.async_state.server = String::from(new_server);
    }

    pub fn enq_verify(&mut self, pkg: &str) {
        let version = self.get_available_debver(pkg);
        match version {
            None => {
                println!("Error: package {} not defined.", pkg);
                return;
            }
            Some(v) => {
                let vs = &v.to_string();
                self.enq_verify_with_version(pkg, vs);
            }
        };
    }

    pub fn enq_verify_with_version(&mut self, pkg: &str, version: &str) {
        let url = format!(
            "http://{}/rest/v1/checksums/{}/{}",
            self.async_state.server, pkg, version
        );

        println!("queueing request {}", url);

        self.async_state
            .reqs
            .push(vec![pkg.to_string(), version.to_string(), url]);
    }

    /// Asks curl to perform all enqueued requests. For requests that succeed with response code 200, compares received MD5sum with local MD5sum (perhaps stored earlier). For requests that fail with 400+, prints error message.
    pub fn execute(&mut self) {
        let mut multi = Multi::new();
        let mut easy_req: Vec<Easy2Handle<Collector>> = Vec::new();

        multi.pipelining(true, true).unwrap();

        for url in self.async_state.reqs.iter() {
            let mut curr_easy = Easy2::new(Collector(Box::new(String::new())));
            curr_easy.url(&url[2]).unwrap();
            curr_easy.verbose(false).unwrap();
            easy_req.push(multi.add2(curr_easy).unwrap());
        }

        while multi.perform().unwrap() > 0 {
            multi.wait(&mut [], Duration::from_secs(10)).unwrap();
        }

        let mut curr_index = 0;
        for el in easy_req.drain(..) {
            let mut handle = multi.remove2(el).unwrap();
            let res_code = handle.response_code().unwrap();
            let res = handle.get_ref().0.to_string();

            if res_code == 200 {
                println!(
                    "verifying {}, matches: {:?}",
                    self.async_state.reqs[curr_index][0],
                    self.md5sums
                        .get(
                            self.package_name_to_num
                                .get(&self.async_state.reqs[curr_index][0])
                                .unwrap()
                        )
                        .unwrap()
                        .eq(&res)
                );
            } else {
                println!(
                    "got error {} on request for package {} version {}",
                    res_code,
                    &self.async_state.reqs[curr_index][0],
                    &self.async_state.reqs[curr_index][1]
                );
            }

            curr_index += 1;
        }
    }
}
