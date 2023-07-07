use dashmap::{DashMap, DashSet};
use regex::Regex;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;
use std::sync::{atomic, Arc, Mutex};
use threadpool::ThreadPool;

use crate::LogFormat;
use crate::LogFormat::Android;
use crate::LogFormat::HealthApp;
use crate::LogFormat::Linux;
use crate::LogFormat::OpenStack;
use crate::LogFormat::Proxifier;
use crate::LogFormat::Spark;
use crate::LogFormat::HDFS;
use crate::LogFormat::HPC;

/**
 * A helper function to track unique thread ids
 * Source: https://stackoverflow.com/a/36115633
 */
static THREAD_COUNT: atomic::AtomicUsize = atomic::AtomicUsize::new(0);
thread_local!(static THREAD_ID: usize = THREAD_COUNT.fetch_add(1, atomic::Ordering::SeqCst));
fn thread_id() -> usize {
    THREAD_ID.with(|&id| id)
}

pub fn format_string(lf: &LogFormat) -> String {
    match lf {
        Linux => r"<Month> <Date> <Time> <Level> <Component>(\\[<PID>\\])?: <Content>".to_string(),
        OpenStack => r"'<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>'"
            .to_string(),
        Spark => r"<Date> <Time> <Level> <Component>: <Content>".to_string(),
        HDFS => r"<Date> <Time> <Pid> <Level> <Component>: <Content>".to_string(),
        HPC => r"<LogId> <Node> <Component> <State> <Time> <Flag> <Content>".to_string(),
        Proxifier => r"[<Time>] <Program> - <Content>".to_string(),
        Android => r"<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>".to_string(),
        HealthApp => "<Time>\\|<Component>\\|<Pid>\\|<Content>".to_string(),
    }
}

pub fn censored_regexps(lf: &LogFormat) -> Vec<Regex> {
    match lf {
        Linux => vec![
            Regex::new(r"(\d+\.){3}\d+").unwrap(),
            Regex::new(r"\w{3} \w{3} \d{2} \d{2}:\d{2}:\d{2} \d{4}").unwrap(),
            Regex::new(r"\d{2}:\d{2}:\d{2}").unwrap(),
        ],
        OpenStack => vec![
            Regex::new(r"((\d+\.){3}\d+,?)+").unwrap(),
            Regex::new(r"/.+?\s").unwrap(),
        ],
        // I commented out Regex::new(r"\d+").unwrap() because that censors all numbers, which may not be what we want?
        Spark => vec![
            Regex::new(r"(\d+\.){3}\d+").unwrap(),
            Regex::new(r"\b[KGTM]?B\b").unwrap(),
            Regex::new(r"([\w-]+\.){2,}[\w-]+").unwrap(),
        ],
        HDFS => vec![
            Regex::new(r"blk_(|-)[0-9]+").unwrap(), // block id
            Regex::new(r"(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)").unwrap(), // IP
        ],
        // oops, numbers require lookbehind, which rust doesn't support, sigh
        //                Regex::new(r"(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$").unwrap()]; // Numbers
        HPC => vec![Regex::new(r"=\d+").unwrap()],
        Proxifier => vec![
            Regex::new(r"<\d+\ssec").unwrap(),
            Regex::new(r"([\w-]+\.)+[\w-]+(:\d+)?").unwrap(),
            Regex::new(r"\d{2}:\d{2}(:\d{2})*").unwrap(),
            Regex::new(r"[KGTM]B").unwrap(),
        ],
        Android => vec![
            Regex::new(r"(/[\w-]+)+").unwrap(),
            Regex::new(r"([\w-]+\.){2,}[\w-]+").unwrap(),
            Regex::new(r"\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b").unwrap(),
        ],
        HealthApp => vec![],
    }
}

// https://doc.rust-lang.org/rust-by-example/std_misc/file/read_lines.html
// The output is wrapped in a Result to allow matching on errors
// Returns an Iterator to the Reader of the lines of the file.
fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

fn regex_generator_helper(format: String) -> String {
    let splitters_re = Regex::new(r"(<[^<>]+>)").unwrap();
    let spaces_re = Regex::new(r" +").unwrap();
    let brackets: &[_] = &['<', '>'];

    let mut r = String::new();
    let mut prev_end = None;
    for m in splitters_re.find_iter(&format) {
        if let Some(pe) = prev_end {
            let splitter = spaces_re.replace(&format[pe..m.start()], r"\s+");
            r.push_str(&splitter);
        }
        let header = m.as_str().trim_matches(brackets).to_string();
        r.push_str(format!("(?P<{}>.*?)", header).as_str());
        prev_end = Some(m.end());
    }
    return r;
}

pub fn regex_generator(format: String) -> Regex {
    return Regex::new(format!("^{}$", regex_generator_helper(format)).as_str()).unwrap();
}

#[test]
fn test_regex_generator_helper() {
    let linux_format =
        r"<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>".to_string();
    assert_eq!(
        regex_generator_helper(linux_format),
        r"(?P<Month>.*?)\s+(?P<Date>.*?)\s+(?P<Time>.*?)\s+(?P<Level>.*?)\s+(?P<Component>.*?)(\[(?P<PID>.*?)\])?:\s+(?P<Content>.*?)"
    );

    let openstack_format =
        r"<Logrecord> <Date> <Time> <Pid> <Level> <Component> (\[<ADDR>\])? <Content>".to_string();
    assert_eq!(
        regex_generator_helper(openstack_format),
        r"(?P<Logrecord>.*?)\s+(?P<Date>.*?)\s+(?P<Time>.*?)\s+(?P<Pid>.*?)\s+(?P<Level>.*?)\s+(?P<Component>.*?)\s+(\[(?P<ADDR>.*?)\])?\s+(?P<Content>.*?)"
    );
}

/// Replaces provided (domain-specific) regexps with <*> in the log_line.
fn apply_domain_specific_re(log_line: String, domain_specific_re: &Vec<Regex>) -> String {
    let mut line = format!(" {}", log_line);
    for s in domain_specific_re {
        line = s.replace_all(&line, "<*>").to_string();
    }
    return line;
}

#[test]
fn test_apply_domain_specific_re() {
    let line = "q2.34.4.5 Jun 14 15:16:02 combo sshd(pam_unix)[19937]: check pass; Fri Jun 17 20:55:07 2005 user unknown".to_string();
    let censored_line = apply_domain_specific_re(line, &censored_regexps(&Linux));
    assert_eq!(
        censored_line,
        " q<*> Jun 14 <*> combo sshd(pam_unix)[19937]: check pass; <*> user unknown"
    );
}

pub fn token_splitter(
    log_line: String,
    re: &Regex,
    domain_specific_re: &Vec<Regex>,
) -> Vec<String> {
    if let Some(m) = re.captures(log_line.trim()) {
        let message = m.name("Content").unwrap().as_str().to_string();
        // println!("{}", &message);
        let line = apply_domain_specific_re(message, domain_specific_re);
        return line
            .trim()
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();
    } else {
        return vec![];
    }
}

#[test]
fn test_token_splitter() {
    let line = "Jun 14 15:16:02 combo sshd(pam_unix)[19937]: check pass; user unknown".to_string();
    let re = regex_generator(format_string(&Linux));
    let split_line = token_splitter(line, &re, &censored_regexps(&Linux));
    assert_eq!(split_line, vec!["check", "pass;", "user", "unknown"]);
}

// processes line, adding to the end of line the first two tokens from lookahead_line, and returns the first 2 tokens on this line
fn process_dictionary_builder_line(
    line: String,
    lookahead_line: Option<String>,
    regexp: &Regex,
    regexps: &Vec<Regex>,
    dbl: &mut HashMap<String, i32>,
    trpl: &mut HashMap<String, i32>,
    all_token_list: &mut Vec<String>,
    prev1: Option<String>,
    prev2: Option<String>,
) {
    let (next1, next2) = match lookahead_line {
        None => (None, None),
        Some(ll) => {
            let next_tokens = token_splitter(ll, &regexp, &regexps);
            match next_tokens.len() {
                0 => (None, None),
                1 => (Some(next_tokens[0].clone()), None),
                _ => (Some(next_tokens[0].clone()), Some(next_tokens[1].clone())),
            }
        }
    };

    let mut tokens = token_splitter(line, &regexp, &regexps);

    if tokens.is_empty() {
        // do nothing
        return;
    }

    tokens.iter().for_each(|t| {
        if !all_token_list.contains(t) {
            all_token_list.push(t.clone())
        }
    });

    let mut tokens2_ = match prev1 {
        None => tokens,
        Some(x) => {
            let mut t = vec![x];
            t.append(&mut tokens);
            t
        }
    };

    let mut tokens2 = match next1 {
        None => tokens2_,
        Some(x) => {
            tokens2_.push(x);
            tokens2_
        }
    };

    for doubles in tokens2.windows(2) {
        let double_tmp = format!("{}^{}", doubles[0], doubles[1]);
        *dbl.entry(double_tmp.to_owned()).or_default() += 1;
    }

    let mut tokens3_ = match prev2 {
        None => tokens2,
        Some(x) => {
            let mut t = vec![x];
            t.append(&mut tokens2);
            t
        }
    };
    let tokens3 = match next2 {
        None => tokens3_,
        Some(x) => {
            tokens3_.push(x);
            tokens3_
        }
    };
    for triples in tokens3.windows(3) {
        let triple_tmp = format!("{}^{}^{}", triples[0], triples[1], triples[2]);
        *trpl.entry(triple_tmp.to_owned()).or_default() += 1;
    }
}

fn process_dictionary_builder_line_dashmap(
    line: String,
    lookahead_line: Option<String>,
    regexp: &Regex,
    regexps: &Vec<Regex>,
    dbl: Arc<DashMap<String, i32>>,
    trpl: Arc<DashMap<String, i32>>,
    all_token_list: Arc<DashSet<String>>,
    prev1: Option<String>,
    prev2: Option<String>,
) {
    let (next1, next2) = match lookahead_line {
        None => (None, None),
        Some(ll) => {
            let next_tokens = token_splitter(ll, &regexp, &regexps);
            match next_tokens.len() {
                0 => (None, None),
                1 => (Some(next_tokens[0].clone()), None),
                _ => (Some(next_tokens[0].clone()), Some(next_tokens[1].clone())),
            }
        }
    };

    let mut tokens = token_splitter(line, &regexp, &regexps);

    if tokens.is_empty() {
        // do nothing
        return;
    }

    tokens.iter().for_each(|t| {
        all_token_list.insert(t.clone());
    });

    let mut tokens2_ = match prev1 {
        None => tokens,
        Some(x) => {
            let mut t = vec![x];
            t.append(&mut tokens);
            t
        }
    };

    let mut tokens2 = match next1 {
        None => tokens2_,
        Some(x) => {
            tokens2_.push(x);
            tokens2_
        }
    };

    for doubles in tokens2.windows(2) {
        let double_tmp = format!("{}^{}", doubles[0], doubles[1]);
        *dbl.entry(double_tmp.to_owned()).or_default() += 1;
    }

    let mut tokens3_ = match prev2 {
        None => tokens2,
        Some(x) => {
            let mut t = vec![x];
            t.append(&mut tokens2);
            t
        }
    };
    let tokens3 = match next2 {
        None => tokens3_,
        Some(x) => {
            tokens3_.push(x);
            tokens3_
        }
    };
    for triples in tokens3.windows(3) {
        let triple_tmp = format!("{}^{}^{}", triples[0], triples[1], triples[2]);
        *trpl.entry(triple_tmp.to_owned()).or_default() += 1;
    }
}

fn find_prev_tokens(
    line: String,
    regexp: &Regex,
    regexps: &Vec<Regex>,
) -> (Option<String>, Option<String>) {
    let tokens = token_splitter(line, &regexp, &regexps);

    (
        match tokens.len() {
            0 => None,
            n => Some(tokens[n - 1].clone()),
        },
        match tokens.len() {
            0 => None,
            1 => None,
            n => Some(tokens[n - 2].clone()),
        },
    )
}

fn merge_hashmaps(maps: Vec<Arc<Mutex<HashMap<String, i32>>>>) -> HashMap<String, i32> {
    // this is for printing when testing
    // for (index, map) in maps.iter().enumerate() {
    //     let locked_map = map.lock().unwrap();
    //     println!("Hashmap {}:", index);
    //     for (key, value) in locked_map.iter() {
    //         println!("\t{}: {}", key, value);
    //     }
    // }

    let mut merged = HashMap::new();
    for map in maps {
        let locked_map = map.lock().unwrap();
        for (key, value) in locked_map.iter() {
            *merged.entry(key.clone()).or_insert(0) += value;
        }
    }
    merged
}

fn merge_token_list(list: Vec<Arc<Mutex<Vec<String>>>>) -> Vec<String> {
    let mut set = HashSet::new();
    for vec in list {
        let locked_vec = vec.lock().unwrap();
        for item in locked_vec.iter() {
            set.insert(item.clone());
        }
    }
    set.into_iter().collect()
}

fn dictionary_builder(
    raw_fn: String,
    format: String,
    regexps: Vec<Regex>,
    num_threads: u32,
    is_single_map: Option<bool>,
) -> (HashMap<String, i32>, HashMap<String, i32>, Vec<String>) {
    let regex = regex_generator(format);
    let (mut prev1, mut prev2) = (None, None);
    let pool = ThreadPool::new(num_threads as usize);

    let final_double_map: HashMap<String, i32>;
    let final_triple_map: HashMap<String, i32>;
    let final_token_list: Vec<String>;

    if is_single_map.is_some() {
        let mut double_maps: Vec<Arc<Mutex<HashMap<String, i32>>>> = vec![];
        let mut triple_maps: Vec<Arc<Mutex<HashMap<String, i32>>>> = vec![];
        let mut token_lists: Vec<Arc<Mutex<Vec<String>>>> = vec![];

        // initialize the three vectors
        for _ in 0..num_threads {
            double_maps.push(Arc::new(Mutex::new(HashMap::new())));
            triple_maps.push(Arc::new(Mutex::new(HashMap::new())));
            token_lists.push(Arc::new(Mutex::new(Vec::new())));
        }

        // iterate through the lines, for each line:
        // 1. get prev1 and prev2, current line, next line, put them into function (processed in threads)
        // 2. update prev1 and prev2 based on the current line
        // 3. move forward
        if let Ok(lines) = read_lines(raw_fn) {
            let mut lp = lines.peekable();

            while let Some(line) = lp.next() {
                // get the value of the current line and the next line
                let curr = line.unwrap();
                let next = match lp.peek() {
                    Some(Ok(value)) => Some(value.clone()),
                    None => None,
                    Some(&Err(_)) => Some("error".to_string()), // should've handle it better, but not important for now.
                };

                // copy variable values so the original doesn't get consumed
                let regexps_clone = regexps.clone();
                let curr_clone = curr.clone();
                let regex_clone = regex.clone();
                let double_maps = double_maps.clone();
                let triple_maps = triple_maps.clone();
                let token_lists = token_lists.clone();

                pool.execute(move || {
                    // get the appropriate hashmap and lock it for safe access
                    let mut double_map = double_maps[thread_id()].lock().unwrap();
                    let mut triple_map = triple_maps[thread_id()].lock().unwrap();
                    let mut all_token_list = token_lists[thread_id()].lock().unwrap();

                    process_dictionary_builder_line(
                        curr_clone,
                        next,
                        &regex_clone,
                        &regexps_clone,
                        &mut double_map,
                        &mut triple_map,
                        &mut all_token_list,
                        prev1,
                        prev2,
                    );
                });

                (prev1, prev2) = find_prev_tokens(curr, &regex, &regexps);
            }

            // wait for all the threads are done
            pool.join();
        }

        // merge all the hashmaps together
        final_double_map = merge_hashmaps(double_maps);
        final_triple_map = merge_hashmaps(triple_maps);
        final_token_list = merge_token_list(token_lists);
    } else {
        let double_map: Arc<DashMap<String, i32>> = Arc::new(DashMap::new());
        let triple_map: Arc<DashMap<String, i32>> = Arc::new(DashMap::new());
        let token_list: Arc<DashSet<String>> = Arc::new(DashSet::new());

        if let Ok(lines) = read_lines(raw_fn) {
            let mut lp = lines.peekable();

            while let Some(line) = lp.next() {
                let curr = line.unwrap();
                let next = match lp.peek() {
                    Some(Ok(value)) => Some(value.clone()),
                    None => None,
                    Some(&Err(_)) => Some("error".to_string()), // should've handle it better, but not important for now.
                };

                let regexps_clone = regexps.clone();
                let curr_clone = curr.clone();
                let regex_clone = regex.clone();
                let double_map = double_map.clone();
                let triple_map: Arc<DashMap<String, i32>> = triple_map.clone();
                let token_list: Arc<DashSet<String>> = token_list.clone();

                pool.execute(move || {
                    process_dictionary_builder_line_dashmap(
                        curr_clone,
                        next,
                        &regex_clone,
                        &regexps_clone,
                        double_map,
                        triple_map,
                        token_list,
                        prev1,
                        prev2,
                    );
                });

                (prev1, prev2) = find_prev_tokens(curr, &regex, &regexps);
            }

            pool.join();
        }

        // cast dashmaps into hashmaps
        final_double_map = Arc::try_unwrap(double_map).unwrap().into_iter().collect();
        final_triple_map = Arc::try_unwrap(triple_map).unwrap().into_iter().collect();
        final_token_list = Arc::try_unwrap(token_list).unwrap().into_iter().collect();
    }

    return (final_double_map, final_triple_map, final_token_list);
}

#[test]
#[ignore]
fn test_dictionary_builder_process_line_lookahead_is_none() {
    let line = "Jun 14 15:16:02 combo sshd(pam_unix)[19937]: check pass; user unknown".to_string();
    let re = regex_generator(format_string(&Linux));
    let mut dbl = HashMap::new();
    let mut trpl = HashMap::new();
    let mut all_token_list = vec![];
    process_dictionary_builder_line(
        line,
        None,
        &re,
        &censored_regexps(&Linux),
        &mut dbl,
        &mut trpl,
        &mut all_token_list,
        None,
        None,
    );
    // assert_eq!(
    //     (last1, last2),
    //     (Some("unknown".to_string()), Some("user".to_string()))
    // );

    let mut dbl_oracle = HashMap::new();
    dbl_oracle.insert("user^unknown".to_string(), 1);
    dbl_oracle.insert("pass;^user".to_string(), 1);
    dbl_oracle.insert("check^pass;".to_string(), 1);
    assert_eq!(dbl, dbl_oracle);

    let mut trpl_oracle = HashMap::new();
    trpl_oracle.insert("pass;^user^unknown".to_string(), 1);
    trpl_oracle.insert("check^pass;^user".to_string(), 1);
    assert_eq!(trpl, trpl_oracle);
}

#[test]
#[ignore]
fn test_dictionary_builder_process_line_lookahead_is_some() {
    let line = "Jun 14 15:16:02 combo sshd(pam_unix)[19937]: check pass; user unknown".to_string();
    let next_line = "Jun 14 15:16:02 combo sshd(pam_unix)[19937]: baz bad".to_string();
    let re = regex_generator(format_string(&Linux));
    let mut dbl = HashMap::new();
    let mut trpl = HashMap::new();
    let mut all_token_list = vec![];
    process_dictionary_builder_line(
        line,
        Some(next_line),
        &re,
        &censored_regexps(&Linux),
        &mut dbl,
        &mut trpl,
        &mut all_token_list,
        Some("foo".to_string()),
        Some("bar".to_string()),
    );

    let mut dbl_oracle = HashMap::new();
    dbl_oracle.insert("unknown^baz".to_string(), 1);
    dbl_oracle.insert("foo^check".to_string(), 1);
    dbl_oracle.insert("user^unknown".to_string(), 1);
    dbl_oracle.insert("pass;^user".to_string(), 1);
    dbl_oracle.insert("check^pass;".to_string(), 1);
    assert_eq!(dbl, dbl_oracle);

    let mut trpl_oracle = HashMap::new();
    trpl_oracle.insert("pass;^user^unknown".to_string(), 1);
    trpl_oracle.insert("check^pass;^user".to_string(), 1);
    trpl_oracle.insert("unknown^baz^bad".to_string(), 1);
    trpl_oracle.insert("foo^check^pass;".to_string(), 1);
    trpl_oracle.insert("bar^foo^check".to_string(), 1);
    trpl_oracle.insert("user^unknown^baz".to_string(), 1);
    assert_eq!(trpl, trpl_oracle);
}

pub fn parse_raw(
    raw_fn: String,
    lf: &LogFormat,
    num_threads: u32,
    is_single_map: Option<bool>,
) -> (HashMap<String, i32>, HashMap<String, i32>, Vec<String>) {
    let (double_dict, triple_dict, all_token_list) = dictionary_builder(
        raw_fn,
        format_string(&lf),
        censored_regexps(&lf),
        num_threads,
        is_single_map,
    );
    println!(
        "double dictionary list len {}, triple {}, all tokens {}",
        double_dict.len(),
        triple_dict.len(),
        all_token_list.len()
    );
    return (double_dict, triple_dict, all_token_list);
}

#[test]
fn test_parse_raw_linux() {
    let (double_dict, triple_dict, all_token_list) =
        parse_raw("data/from_paper.log".to_string(), &Linux, 8, false);
    let all_token_list_oracle = vec![
        "hdfs://hostname/2kSOSP.log:21876+7292".to_string(),
        "hdfs://hostname/2kSOSP.log:14584+7292".to_string(),
        "hdfs://hostname/2kSOSP.log:0+7292".to_string(),
        "hdfs://hostname/2kSOSP.log:7292+7292".to_string(),
        "hdfs://hostname/2kSOSP.log:29168+7292".to_string(),
    ];
    assert_eq!(all_token_list, all_token_list_oracle);
    let mut double_dict_oracle = HashMap::new();
    double_dict_oracle.insert(
        "hdfs://hostname/2kSOSP.log:14584+7292^hdfs://hostname/2kSOSP.log:0+7292".to_string(),
        2,
    );
    double_dict_oracle.insert(
        "hdfs://hostname/2kSOSP.log:21876+7292^hdfs://hostname/2kSOSP.log:14584+7292".to_string(),
        2,
    );
    double_dict_oracle.insert(
        "hdfs://hostname/2kSOSP.log:7292+7292^hdfs://hostname/2kSOSP.log:29168+7292".to_string(),
        2,
    );
    double_dict_oracle.insert(
        "hdfs://hostname/2kSOSP.log:0+7292^hdfs://hostname/2kSOSP.log:7292+7292".to_string(),
        2,
    );
    assert_eq!(double_dict, double_dict_oracle);
    let mut triple_dict_oracle = HashMap::new();
    triple_dict_oracle.insert("hdfs://hostname/2kSOSP.log:0+7292^hdfs://hostname/2kSOSP.log:7292+7292^hdfs://hostname/2kSOSP.log:29168+7292".to_string(), 1);
    triple_dict_oracle.insert("hdfs://hostname/2kSOSP.log:14584+7292^hdfs://hostname/2kSOSP.log:0+7292^hdfs://hostname/2kSOSP.log:7292+7292".to_string(), 1);
    triple_dict_oracle.insert("hdfs://hostname/2kSOSP.log:21876+7292^hdfs://hostname/2kSOSP.log:14584+7292^hdfs://hostname/2kSOSP.log:0+7292".to_string(), 1);
    assert_eq!(triple_dict, triple_dict_oracle);
}

/// standard mapreduce invert map: given {<k1, v1>, <k2, v2>, <k3, v1>}, returns ([v1, v2] (sorted), {<v1, [k1, k3]>, <v2, [k2]>})
pub fn reverse_dict(d: &HashMap<String, i32>) -> (BTreeSet<i32>, HashMap<i32, Vec<String>>) {
    let mut reverse_d: HashMap<i32, Vec<String>> = HashMap::new();
    let mut val_set: BTreeSet<i32> = BTreeSet::new();

    for (key, val) in d.iter() {
        if reverse_d.contains_key(val) {
            let existing_keys = reverse_d.get_mut(val).unwrap();
            existing_keys.push(key.to_string());
        } else {
            reverse_d.insert(*val, vec![key.to_string()]);
            val_set.insert(*val);
        }
    }
    return (val_set, reverse_d);
}

pub fn print_dict(s: &str, d: &HashMap<String, i32>) {
    let (val_set, reverse_d) = reverse_dict(d);

    println!("printing dict: {}", s);
    for val in &val_set {
        println!("{}: {:?}", val, reverse_d.get(val).unwrap());
    }
    println!("---");
}
