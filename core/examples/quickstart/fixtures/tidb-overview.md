---
title: What is TiDB Self-Managed
summary: Learn about the key features and usage scenarios of TiDB.
aliases: ['/docs/dev/key-features/','/tidb/dev/key-features','/docs/dev/overview/']
---

# What is TiDB Self-Managed

<!-- Localization note for TiDB:

- English: use distributed SQL, and start to emphasize HTAP
- Chinese: can keep "NewSQL" and emphasize one-stop real-time HTAP ("一栈式实时 HTAP")
- Japanese: use NewSQL because it is well-recognized

-->

[TiDB](https://github.com/pingcap/tidb) (/'taɪdiːbi:/, "Ti" stands for Titanium) is an open-source distributed SQL database that supports Hybrid Transactional and Analytical Processing (HTAP) workloads. It is MySQL compatible and features horizontal scalability, strong consistency, and high availability. The goal of TiDB is to provide users with a one-stop database solution that covers OLTP (Online Transactional Processing), OLAP (Online Analytical Processing), and HTAP services. TiDB is suitable for various use cases that require high availability and strong consistency with large-scale data.

TiDB Self-Managed is a product option of TiDB, where users or organizations can deploy and manage TiDB on their own infrastructure with complete flexibility. With TiDB Self-Managed, you can enjoy the power of open source, distributed SQL while retaining full control over your environment.

The following video introduces key features of TiDB.

<iframe width="600" height="450" src="https://www.youtube.com/embed/aWBNNPm21zg?enablejsapi=1" title="Why TiDB?" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Key features

- **Easy horizontal scaling**

  The TiDB architecture design separates computing from storage, letting you scale out or scale in the computing or storage capacity online as needed. The scaling process is transparent to application operations and maintenance staff.

- **Financial-grade high availability**

  Data is stored in multiple replicas, and the Multi-Raft protocol is used to obtain the transaction log. A transaction can only be committed when data has been successfully written into the majority of replicas. This guarantees strong consistency and availability when a minority of replicas go down. You can configure the geographic location and number of replicas as needed to meet different disaster tolerance levels.

- **Real-time HTAP**

  TiDB provides two storage engines: [TiKV](/tikv-overview.md), a row-based storage engine, and [TiFlash](/tiflash/tiflash-overview.md), a columnar storage engine. 

  TiFlash uses the Multi-Raft Learner protocol to replicate data from TiKV in real time, ensuring consistent data between the TiKV row-based storage engine and the TiFlash columnar storage engine. TiKV and TiFlash can be deployed on different machines as needed to solve the problem of HTAP resource isolation.

- **Cloud-native distributed database**

  TiDB is a distributed database designed for the cloud, providing flexible scalability, reliability, and security on the cloud platform. Users can elastically scale TiDB to meet the requirements of their changing workloads. In TiDB, each piece of data has at least 3 replicas, which can be scheduled in different cloud availability zones to tolerate the outage of a whole data center. [TiDB Operator](https://docs.pingcap.com/tidb-in-kubernetes/stable/tidb-operator-overview) helps manage TiDB on Kubernetes and automates tasks related to operating the TiDB cluster, making TiDB easier to deploy on any cloud that provides managed Kubernetes. [TiDB Cloud](https://pingcap.com/tidb-cloud/), the fully-managed TiDB service, is the easiest, most economical, and most resilient way to unlock the full power of [TiDB in the cloud](https://docs.pingcap.com/tidbcloud/), allowing you to deploy and run TiDB clusters with just a few clicks.

- **Compatible with the MySQL protocol and MySQL ecosystem**

  TiDB is compatible with the MySQL protocol, common features of MySQL, and the MySQL ecosystem. To migrate applications to TiDB, you do not need to change a single line of code in many cases, or only need to modify a small amount of code. In addition, TiDB provides a series of [data migration tools](/ecosystem-tool-user-guide.md) to help easily migrate application data into TiDB.

## See also

- [TiDB Architecture](/tidb-architecture.md)
- [TiDB Storage](/tidb-storage.md)
- [TiDB Computing](/tidb-computing.md)
- [TiDB Scheduling](/tidb-scheduling.md)
