Title: An R Function for Generating Authenticated URLs to Private Web Sites Hosted on AWS S3
Date: 2016-09-19
Tags: AWS

![crypto][hmacWiki]

Quite often I want to share simple (static) web pages with other colleagues or clients. For example, I may have written a report using [R Markdown][rmarkdown] and rendered it to HTML. AWS S3 can easily host such a simple web page (e.g. see [here][awsStaticS3]), but it cannot, however, offer any authentication to prevent anyone from accessing potentially sensitive information.

Yegor Bugayenko has created an external service [S3Auth.com][s3AuthDotCom] that stands in the way of any S3 hosted web site, but this is a little too much for my needs. All I want to achieve is to limit access to specific S3 resources that will be largely transient in nature. A viable and simple solution is to use 'query string request authentication' that is described in detail [here][queryStringReqAuth]. I must confess to not really understanding what was going on here, until I had dug around on the web to see what others have been up to.

This blog post describes a simple R function for generating authenticated and ephemeral URLs to private S3 resources (including web pages) that only the holders of the URL can access.

# Creating User Credentials for Read-Only Access to S3

Before we can authenticate anyone, we need someone to authenticate. From the AWS Management Console create a new user, download their security credentials and then attach the `AmazonS3ReadOnlyAccess` policy to them. For more details on how to do this, refer to a [previous post][partOne]. Note, that you should **not** create passwords for them to access the AWS console.

# Loading a Static Web Page to AWS S3

Do **not** be tempted to follow the S3 'Getting Started' page on how to host a static web page and in doing so enable 'Static Website Hosting'. We need our resources to remain private and we would also like to use HTTPS, which this option does not support. Instead, create a new bucket and upload a simple HTML file [as usual][partOne]. An example html file - e.g. `index.html` - could be,

```html
<!DOCTYPE html>
<html>
  <body>
    <p>Hello, World!</p>
  </body>
</html>
```

# An R Function for Generating Authenticated URLs
We can now use our new user's Access Key ID and Secret Access Key to create a URL with a limited lifetime that enables access to `index.html`. Technically, we are making a HTTP GET request to the S3 REST API, with the authentication details sent as part of a query string. Creating this URL is a bit tricky - I have adapted the Python example (number 3) that is provided [here][pythonExample], as an R function (that can be found in the Gist below) - `aws_query_string_auth_url(...)`. Here's an example showing this R function in action:

```r
path_to_file <- "index.html"
bucket <- "my.s3.bucket"
region <- "eu-west-1"
aws_access_key_id <- "DWAAAAJL4KIEWJCV3R36"
aws_secret_access_key <- "jH1pEfnQtKj6VZJOFDy+t253OZJWZLEo9gaEoFAY"
lifetime_minutes <- 1
aws_query_string_auth_url(path_to_file, bucket, region, aws_access_key_id, aws_secret_access_key, lifetime_minutes)
# "https://s3-eu-west-1.amazonaws.com/my.s3.bucket/index.html?AWSAccessKeyId=DWAAAKIAJL4EWJCV3R36&Expires=1471994487&Signature=inZlnNHHswKmcPfTBiKhziRSwT4%3D"
```

And here's the code for it as inspired by the short code snippet [here][pythonExample]:

<script src="https://gist.github.com/AlexIoannides/927dc77c8258ab436f602096c8491460.js"></script>

Note the dependencies on the `digest` and `base64enc` packages.

[hmacWiki]: https://alexioannides.files.wordpress.com/2016/08/hmac.png "HMAC"

[rmarkdown]: http://rmarkdown.rstudio.com "R Markdown @ R Studio"

[awsStaticS3]: http://docs.aws.amazon.com/gettingstarted/latest/swh/website-hosting-intro.html "AWS S3 Static Web Page"

[queryStringReqAuth]: http://docs.aws.amazon.com/AmazonS3/latest/dev/RESTAuthentication.html#RESTAuthenticationQueryStringAuth "AWS documentation"

[s3AuthDotCom]: http://www.s3auth.com "S3 Authentication Service"

[partOne]: https://alexioannides.com/2016/08/16/building-a-data-science-platform-for-rd-part-1-setting-up-aws/ "Part 1"

[pythonExample]: https://s3.amazonaws.com/doc/s3-developer-guide/RESTAuthentication.html "Python Auth Example"
