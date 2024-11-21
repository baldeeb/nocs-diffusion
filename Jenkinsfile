pipeline {
  agent any
  stages {
    stage('First Stage') {
      steps {
        echo 'Started!'
        sh 'sh python ./scripts/train_nocs_diffusion.py'
        echo 'Done!'
      }
    }

    stage('Second Stage') {
      steps {
        echo 'Step two reached!!!!'
      }
    }

  }
}