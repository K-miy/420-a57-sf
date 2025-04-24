FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Toronto
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update -y \
    && apt-get -y install build-essential software-properties-common \
    && apt-get -y install gdebi-core libcurl4-gnutls-dev libxml2-dev libssl-dev \
    && apt-get -y install ca-certificates

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9

RUN add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran35/'

RUN apt-get update -y \
    && apt-get -y install r-base \
    && apt-get -y install wget curl

RUN mkdir /tmp/rstudio.deb

RUN wget https://download2.rstudio.org/server/bionic/amd64/rstudio-server-1.2.5019-amd64.deb -P /tmp/rstudio.deb

RUN ln -s /dev/stdout /var/log/syslog \
      && apt-get -y install /tmp/rstudio.deb/rstudio-server-1.2.5019-amd64.deb

RUN useradd -m -d /home/rstudio -g rstudio-server rstudio \
      && echo rstudio:rstudio | chpasswd \
      && echo "r-cran-repos=${CRAN_URL}" >> /etc/rstudio/rsession.conf

EXPOSE 8787

RUN Rscript -e "install.packages('ggplot2')"
RUN Rscript -e "install.packages('devtools')"
RUN Rscript -e "install.packages('mapproj')"

RUN R -e "devtools::install_github('wmurphyrd/fiftystater')"

ENTRYPOINT ["/usr/lib/rstudio-server/bin/rserver"]
CMD ["--server-daemonize=0", "--server-app-armor-enabled=0"]
