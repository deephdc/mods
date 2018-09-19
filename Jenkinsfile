node {
  def githubcredentials = '_(CHANGE!)_' // github credentials as stored in Jenkins
  def dockerhubuser = 'sht3v0'
  def dockerhubcredentials = '_(CHANGE!)_' // dockerhub credentials as stored in Jenkins
  def appName = 'mods'
  def mainVer = '0.0.1'
  def imageTagBase = "${appName}:${env.BRANCH_NAME}-${mainVer}.${env.BUILD_NUMBER}"
  def imageTagExtension = ''    // e.g. '-gpu'
  def imageTag = "${dockerhubuser}/deep-oc-${imageTagBase}${imageTagExtension}"
  def imageTagLatest = "${dockerhubuser}/deep-oc-${appName}"
  
  try {
      stage ('Clone repositories') {
          dir("${appName}") {
              checkout([$class: 'GitSCM', branches: [[name: '*/master']],
                  credentialsId: "${githubcredentials}",
                  userRemoteConfigs: [[url: "http://github.com/Stifo/${appName}.git"]]])
          }

          dir("DEEP-OC-${appName}") {
              checkout([$class: 'GitSCM', branches: [[name: '*/master']],
                  credentialsId: "${githubcredentials}",
                  userRemoteConfigs: [[url: "http://github.com/Stifo/DEEP-OC-${appName}.git"]]])
          }
      }

      stage ('Build test image and run tests') {
          dir("${appName}") {
              def imageTagTest = "${imageTagBase}-tests"
              sh("nvidia-docker build -t ${imageTagTest} -f docker/Dockerfile.tests .")
              sh("docker run ${imageTagTest} ./run_pylint.sh >pylint.log || exit 0")        
              warnings canComputeNew: false, canResolveRelativePaths: false, categoriesPattern: '', defaultEncoding: '', excludePattern: '', healthy: '', includePattern: '', messagesPattern: '', parserConfigurations: [[parserName: 'PyLint', pattern: '**/pylint.log']], unHealthy: ''

              echo "Here should be more tests for ${imageTagTest}"
              // delete test docker image from Jenkins site
              sh("docker rmi --force ${imageTagTest}")
          }
      }


      stage ('Build and Push docker image to registry') {
          dir("DEEP-OC-${appName}") { 
              sh("nvidia-docker build -t ${imageTag} -t ${imageTagLatest} .")
              withCredentials([usernamePassword(credentialsId: "${dockerhubcredentials}", usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD')]) {
                  sh '''
                     docker login -u "$USERNAME" -p "$PASSWORD"
                     '''
              }
              sh("docker push ${imageTag}")
              sh("docker push ${imageTagLatest}")  
          }    
      }

      stage('Deploy Application') {
      }

      stage('Post Deployment') {
          // delete docker image from Jenkins site
          sh("docker rmi ${imageTag}")
          sh("docker rmi ${imageTagLatest}")
      }
  } catch (e) {
    // If there was an exception thrown, the build failed
    currentBuild.result = "FAILED"
    throw e
  } finally {
    // Success or failure, always send notifications
    notifyBuild()
  }
}

def notifyBuild() {
    String buildStatus =  currentBuild.result
    // build status of null means successful
    buildStatus =  buildStatus ?: 'SUCCESS'
  
    // One can re-define default values
    def subject = "${buildStatus}: Job '${env.JOB_NAME} [${env.BUILD_NUMBER}]'"
    def summary = "${subject} (${env.BUILD_URL})"
    def details = """<p>STARTED: Job '${env.JOB_NAME} - build # ${env.BUILD_NUMBER}' on $env.NODE_NAME.</p>
      <p>TERMINATED with: ${buildStatus}
      <p>Check console output at "<a href="${env.BUILD_URL}">${env.BUILD_URL}</a>"</p>"""


    emailext (
        subject: '${DEFAULT_SUBJECT}', //subject,
        mimeType: 'text/html',
        body: details,                 //'${DEFAULT_CONTENT}'
        attachLog: true,
        compressLog: true,
        attachmentsPattern: '**/pylint.log',
        recipientProviders: [[$class: 'CulpritsRecipientProvider'],
                            [$class: 'DevelopersRecipientProvider'],
                            [$class: 'RequesterRecipientProvider']]
    )
}
