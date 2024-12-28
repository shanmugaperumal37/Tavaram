plugins {
    id("com.android.application") version "7.4.1" apply false
    id("org.jetbrains.kotlin.android") version "1.8.10" apply false
}

task("clean", type = Delete::class) {
    delete(rootProject.buildDir)
}
