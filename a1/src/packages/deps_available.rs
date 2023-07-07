use crate::packages::Dependency;
use crate::packages::RelVersionedPackageNum;
use crate::Packages;
use rpkg::debversion::{self, DebianVersionNum};

impl Packages {
    /// Gets the dependencies of package_name, and prints out whether they are satisfied (and by which library/version) or not.
    pub fn deps_available(&self, package_name: &str) {
        if !self.package_exists(package_name) {
            println!("no such package {}", package_name);
            return;
        }

        println!("Package {}:", package_name);
        let deps: &Vec<Dependency> = self
            .dependencies
            .get(self.get_package_num(package_name))
            .unwrap();

        for dd in deps {
            println!("- dependency \"{}\"", self.dep2str(dd));

            if let Some((pkg, ver)) = self.dep_is_satisfied(dd) {
                println!("+ {} satisfied by installed version {}", pkg, ver);
            } else {
                println!("-> not satisfied");
            }
        }
    }

    /// Returns Some(package) which satisfies dependency dd, or None if not satisfied.
    pub fn dep_is_satisfied(&self, dd: &Dependency) -> Option<(&str, &DebianVersionNum)> {
        for d in dd {
            let package_name = self.get_package_name(d.package_num);
            let installed_version = self.get_installed_debver(package_name);

            if installed_version.is_none() {
                continue; // no such package installed
            } else {
                // package is installed, check version number
                let installed_version = installed_version.unwrap();

                // if no version number is specified, then the package is satisfied
                if d.rel_version.is_none() {
                    return Some((package_name, installed_version));
                } else {
                    let rel_version = d.rel_version.as_ref().unwrap();
                    let op = &rel_version.0;
                    let ver = &rel_version.1;
                    let ver_debver = ver.parse::<debversion::DebianVersionNum>().unwrap();

                    if debversion::cmp_debversion_with_op(op, installed_version, &ver_debver) {
                        return Some((package_name, installed_version));
                    } else {
                        continue;
                    }
                }
            }
        }
        return None;
    }

    /// Returns a Vec of packages which would satisfy dependency dd but for the version.
    /// Used by the how-to-install command, which calls compute_how_to_install().
    pub fn _dep_satisfied_by_wrong_version(&self, dd: &Dependency) -> Vec<&str> {
        assert!(self.dep_is_satisfied(dd).is_none());
        let mut result = vec![];

        for d in dd {
            let package_name = self.get_package_name(d.package_num);
            let installed_version = self.get_installed_debver(package_name);

            if installed_version.is_none() {
                continue;
            } else {
                // package is installed, check version number
                let installed_version = installed_version.unwrap();

                if d.rel_version.is_none() {
                    continue;
                } else {
                    let rel_version = d.rel_version.as_ref().unwrap();
                    let op = &rel_version.0;
                    let ver = &rel_version.1;
                    let ver_debver = ver.parse::<debversion::DebianVersionNum>().unwrap();

                    if debversion::cmp_debversion_with_op(op, installed_version, &ver_debver) {
                        continue; // it does have the right version
                    } else {
                        result.push(package_name); // TODO: check if this function works
                    }
                }
            }
        }

        return result;
    }

    pub fn satisfied_by_wrong_version(&self, d: &RelVersionedPackageNum) -> bool {
        let package_name = self.get_package_name(d.package_num);
        let installed_version = self.get_installed_debver(package_name);

        if installed_version.is_none() {
            false
        } else {
            // package is installed, check version number
            let installed_version = installed_version.unwrap();

            if d.rel_version.is_none() {
                false
            } else {
                let rel_version = d.rel_version.as_ref().unwrap();
                let op = &rel_version.0;
                let ver = &rel_version.1;
                let ver_debver = ver.parse::<debversion::DebianVersionNum>().unwrap();

                !debversion::cmp_debversion_with_op(op, installed_version, &ver_debver)
            }
        }
    }

    // this is a duplicate of dep2str :(
    pub fn _dep_to_str(&self, dd: &Dependency) -> String {
        let mut result = String::from("\"");
        let mut first = true;
        for d in dd {
            if !first {
                result.push_str(" | ");
            }
            result.push_str(&format!(
                "{}{}",
                self.get_package_name(d.package_num),
                match d.rel_version.as_ref() {
                    None => String::new(),
                    Some((version_relation, version_number)) =>
                        format!(" ({} {})", version_relation.to_string(), version_number),
                }
            ));
            first = false;
        }
        result.push_str("\"");
        return result;
    }
}
