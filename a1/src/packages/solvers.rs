use std::str::FromStr;

use rpkg::debversion::DebianVersionNum;

use crate::debversion;
use crate::packages::Dependency;
use crate::Packages;

impl Packages {
    pub fn transitive_dep_solution(&self, package_name: &str) -> Vec<i32> {
        if !self.package_exists(package_name) {
            return vec![];
        }

        let deps: &Vec<Dependency> = &*self
            .dependencies
            .get(self.get_package_num(package_name))
            .unwrap();

        let (mut dependency_set, mut curr_index, mut additional_index) = (vec![], 0, 0);
        let mut additional_deps: Vec<&Dependency> = vec![];

        while curr_index < deps.len() || additional_index < additional_deps.len() {
            let item = if curr_index < deps.len() {
                curr_index += 1;
                &deps[curr_index - 1][0]
            } else {
                additional_index += 1;
                &additional_deps[additional_index - 1][0]
            };

            if !dependency_set.contains(&item.package_num) {
                // add the current item to the dependency list
                dependency_set.push(item.package_num);

                // add all its dependency to be processed in the future
                let more_deps = self.dependencies.get(&item.package_num);
                if more_deps.is_some() {
                    for md in more_deps.unwrap() {
                        additional_deps.push(md);
                    }
                }
            }
        }

        return dependency_set;
    }

    pub fn compute_how_to_install(&self, package_name: &str) -> Vec<i32> {
        if !self.package_exists(package_name) {
            return vec![];
        }
        let mut dependencies_to_add: Vec<i32> = vec![];

        let deps: &Vec<Dependency> = &*self
            .dependencies
            .get(self.get_package_num(package_name))
            .unwrap();

        let (mut deps_iter, mut deps_to_add_iter) = (0, 0);
        let mut more_deps: Vec<&Dependency> = vec![];

        while deps_iter < deps.len() || deps_to_add_iter < more_deps.len() {
            let d = if deps_iter < deps.len() {
                deps_iter += 1;
                &deps[deps_iter - 1]
            } else {
                deps_to_add_iter += 1;
                more_deps[deps_to_add_iter - 1]
            };

            if self.dep_is_satisfied(d).is_some() {
                continue;
            }

            if d.len() == 1 {
                if !dependencies_to_add.contains(&d[0].package_num) {
                    dependencies_to_add.push(d[0].package_num);
                    for md in self.dependencies.get(&d[0].package_num).unwrap() {
                        more_deps.push(md);
                    }
                }
            } else {
                let (mut num_of_wrong_version, mut index_of_wrong_version) = (0, 0);

                for (i, el) in d.iter().enumerate() {
                    if self.satisfied_by_wrong_version(el) {
                        num_of_wrong_version += 1;
                        index_of_wrong_version = i;
                    }
                }

                if num_of_wrong_version == 1 {
                    // only one with wrong version, install that one
                    if !dependencies_to_add.contains(&d[index_of_wrong_version].package_num) {
                        dependencies_to_add.push(d[index_of_wrong_version].package_num);
                        for md in self
                            .dependencies
                            .get(&d[index_of_wrong_version].package_num)
                            .unwrap()
                        {
                            more_deps.push(md);
                        }
                    }
                } else {
                    // between the ones with the wrong version number installed,
                    // choose the one with the highest version number.
                    let (mut index_of_the_highest_ver, mut highest_ver) =
                        (0, DebianVersionNum::from_str("1.0").unwrap());

                    for (i, el) in d.iter().enumerate() {
                        if self.satisfied_by_wrong_version(el) {
                            // get the version number
                            match &el.rel_version {
                                None => (),
                                Some((_op, ver)) => {
                                    let curr_debver =
                                        ver.parse::<debversion::DebianVersionNum>().unwrap();

                                    if debversion::cmp_debversion_with_op(
                                        &debversion::VersionRelation::StrictlyGreater,
                                        &highest_ver,
                                        &curr_debver,
                                    ) {
                                        index_of_the_highest_ver = i;
                                        highest_ver = curr_debver;
                                    }
                                }
                            };
                        }
                    }

                    if !dependencies_to_add.contains(&d[index_of_the_highest_ver].package_num) {
                        dependencies_to_add.push(d[index_of_the_highest_ver].package_num);
                        for md in self
                            .dependencies
                            .get(&d[index_of_the_highest_ver].package_num)
                            .unwrap()
                        {
                            more_deps.push(md);
                        }
                    }
                }
            }
        }

        return dependencies_to_add;
    }
}
